#!/usr/bin/env python3
"""Time the NEGF solve on CPU vs GPU across device sizes.

Each configuration runs in fresh subprocesses -- one with VIDES_GPU=0 and
one with VIDES_GPU=1 -- because the backend is chosen once per process.
Inside each, the same charge/transmission call is made `--repeat` times:

  first   the first call.  On the GPU it includes one-off CUDA context and
          cuBLAS start-up, which a long self-consistent run pays only once.
  steady  the best of the remaining calls: what each further NEGF solve in
          a self-consistent loop costs.

Speedup is CPU steady / GPU steady.

    cd src && PYTHONPATH=. python3 ../test/benchmark_gpu.py
    cd src && PYTHONPATH=. python3 ../test/benchmark_gpu.py --quick
    cd src && PYTHONPATH=. python3 ../test/benchmark_gpu.py --case gnr --repeat 3

The C code prints progress for every solve; it is captured and discarded,
so only the table is shown.  Large cases take minutes on the CPU side --
use --quick for a first look.

Results are also saved to benchmark_results.json (see --out) for
test/plot_benchmark.py.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------
# Device builders.  Each returns an object with charge_T() and n, Nc.
# Energy window and step are fixed per case so NE is the same on both
# backends; the grid is offset by half a step to stay off band-centre
# resonances (see test_gpu_vs_cpu.py, case hamiltonian).
# ----------------------------------------------------------------------

def build_cnt(size):
    from NanoTCAD_ViDES import nanotube
    chirality, length = size
    d = nanotube(chirality, length)
    d.Elower, d.Eupper, d.dE = -1.505, 1.505, 0.005
    d.Phi = -0.2 * np.ones(d.n * d.Nc)
    d.mu2 = -0.3
    return d


def build_gnr(size):
    from NanoTCAD_ViDES import nanoribbon
    width, length = size
    d = nanoribbon(width, length)
    d.Elower, d.Eupper, d.dE = -1.505, 1.505, 0.005
    d.Phi = -0.2 * d.z
    d.mu2 = -0.3
    return d


def build_hamiltonian(size):
    from NanoTCAD_ViDES import Hamiltonian
    sys.path.insert(0, os.path.join(HERE, "..", "demo"))
    from GNR import GNR
    width, slices = size
    d = Hamiltonian(width, slices)
    d.H = GNR(width, slices)
    d.Elower, d.Eupper, d.dE = -1.505, 1.505, 0.005
    d.eta = 1e-5
    d.Phi = -0.2 * np.ones(d.n * d.Nc)
    d.mu2 = -0.3
    return d


# (label, builder, sizes, quick sizes).  Sizes grow along the axes that
# matter: block size n (width / chirality) and number of blocks Nc
# (length).  ~600 energy points per solve.
CASES = {
    "cnt": ("CNT (n,0), length nm", build_cnt,
            [(10, 5), (10, 20), (19, 20), (25, 40)],
            [(10, 5), (19, 20)]),
    "gnr": ("GNR width, length nm", build_gnr,
            [(6, 5), (12, 10), (24, 10), (24, 30)],
            [(6, 5), (12, 10)]),
    "hamiltonian": ("Hamiltonian (Lake), width x slices", build_hamiltonian,
                    [(6, 20), (12, 40), (24, 40), (24, 120)],
                    [(6, 20), (12, 40)]),
}


# ----------------------------------------------------------------------
# Worker: one backend, one size, `repeat` timed solves.
# ----------------------------------------------------------------------

def worker(case, size, repeat, out):
    builder = CASES[case][1]
    times = []
    d = None
    for _ in range(repeat):
        d = builder(size)          # fresh device each time, same work
        t0 = time.perf_counter()
        d.charge_T()
        times.append(time.perf_counter() - t0)
    # Count energy points from the grid.  len(d.E) is not it: some paths
    # (H_charge_T) fill the start of a preallocated NEmax-long array.
    ne = int(round((d.Eupper - d.Elower) / d.dE)) + 1
    info = {"times": times, "n": int(d.n), "Nc": int(d.Nc), "NE": ne}
    with open(out, "w") as f:
        json.dump(info, f)


def run(case, size, use_gpu, repeat, tmpdir):
    out = os.path.join(tmpdir, "r_%s_%s_%d.json"
                       % (case, "x".join(map(str, size)), use_gpu))
    env = dict(os.environ)
    env["VIDES_GPU"] = "1" if use_gpu else "0"
    env["VIDES_PROFILE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [os.getcwd(), env.get("PYTHONPATH", "")] if p)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--worker",
         case, json.dumps(list(size)), str(repeat), out],
        env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-2000:])
        return None, ""
    backend = ""
    prof = []
    for line in proc.stdout.splitlines():
        if "NEGF backend" in line and not backend:
            backend = line.strip()
        m = PROFILE_RE.search(line)
        if m:
            prof.append(tuple(float(g) for g in m.groups()))
    with open(out) as f:
        info = json.load(f)
    # Keep the last solve's split, which matches the steady-state time.
    if prof:
        info["prof_total"], info["prof_self"], info["prof_solve"] = prof[-1]
    return info, backend


PROFILE_RE = re.compile(r"\[ViDES profile\].*total ([0-9.eE+-]+) s = "
                        r"self-energy ([0-9.eE+-]+) s.*NEGF solve "
                        r"([0-9.eE+-]+) s")


def share(info, key):
    t = info.get("prof_total")
    return 100.0 * info[key] / t if t else float("nan")


def fmt(t):
    return "%.2f s" % t if t >= 0.1 else "%.0f ms" % (1e3 * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=4, help=argparse.SUPPRESS)
    ap.add_argument("--case", choices=sorted(CASES), action="append",
                    help="benchmark only this path (repeatable)")
    ap.add_argument("--repeat", type=int, default=2,
                    help="solves per backend per size (default 2: one "
                         "first call, one steady-state)")
    ap.add_argument("--out", default="benchmark_results.json",
                    help="where to save the results for plot_benchmark.py "
                         "(default: benchmark_results.json)")
    ap.add_argument("--quick", action="store_true",
                    help="two small sizes per path only")
    args = ap.parse_args()

    if args.worker:
        case, size, repeat, out = args.worker
        worker(case, tuple(json.loads(size)), int(repeat), out)
        return 0

    import tempfile
    tmp = tempfile.mkdtemp(prefix="vides_bench_")
    repeat = max(1, args.repeat)
    rows = []
    gpu_line = None

    for case in (args.case or ["cnt", "gnr", "hamiltonian"]):
        label, _, sizes, quick = CASES[case]
        for size in (quick if args.quick else sizes):
            tag = "%s %s" % (case, "x".join(map(str, size)))
            print("running %-22s" % tag, end="", flush=True)
            # The CPU has no start-up cost to separate out, so one fewer
            # solve there: the slow side of the benchmark stays short.
            cpu, _ = run(case, size, False, max(1, repeat - 1), tmp)
            gpu, line = run(case, size, True, repeat, tmp)
            if cpu is None or gpu is None:
                print("  failed (output above)")
                continue
            if line and gpu_line is None:
                gpu_line = line
            c_steady = min(cpu["times"][1:] or cpu["times"])
            g_first = gpu["times"][0]
            g_steady = min(gpu["times"][1:] or gpu["times"])
            rows.append((case, size, cpu["n"], cpu["Nc"], cpu["NE"],
                         c_steady, g_first, g_steady, c_steady / g_steady,
                         share(cpu, "prof_self") if "prof_self" in cpu else float("nan"),
                         share(gpu, "prof_self") if "prof_self" in gpu else float("nan")))
            print("  done")

    print()
    if gpu_line:
        print(gpu_line)
        if "no CUDA device" in gpu_line or "disabled" in gpu_line:
            print("WARNING: the 'GPU' runs did not use a GPU; the speedups "
                  "below are meaningless.")
    print()
    hdr = ("%-12s %-9s %5s %5s %5s  %10s  %10s %10s  %8s  %8s %8s"
           % ("path", "size", "n", "Nc", "NE", "CPU", "GPU first",
              "GPU steady", "speedup", "CPU:Σ%", "GPU:Σ%"))
    print(hdr)
    print("-" * len(hdr))
    for case, size, n, Nc, NE, c, gf, gs, sp, sc, sg in rows:
        print("%-12s %-9s %5d %5d %5d  %10s  %10s %10s  %7.1fx  %7.0f%% %7.0f%%"
              % (case, "x".join(map(str, size)), n, Nc, NE,
                 fmt(c), fmt(gf), fmt(gs), sp, sc, sg))
    print()
    print("CPU = best CPU solve; GPU steady = best GPU solve after the first. "
          "Speedup = CPU / GPU steady.")
    print("n = block size, Nc = number of blocks, NE = energy points per solve.")
    print("Σ% = share of the solve spent computing contact self-energies "
          "(batched on the GPU for GNR and the Hamiltonian path; analytical, "
          "per energy on the CPU, for CNTs).")

    result = {
        "backend": gpu_line or "",
        "repeat": repeat,
        "rows": [
            {"path": case, "size": list(size), "n": n, "Nc": Nc, "NE": NE,
             "cpu_s": c, "gpu_first_s": gf, "gpu_steady_s": gs,
             "speedup": sp, "cpu_selfenergy_pct": sc,
             "gpu_selfenergy_pct": sg}
            for case, size, n, Nc, NE, c, gf, gs, sp, sc, sg in rows
        ],
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("Saved %s -- plot it with: python3 ../test/plot_benchmark.py %s"
          % (args.out, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
