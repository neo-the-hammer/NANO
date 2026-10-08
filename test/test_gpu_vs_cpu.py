#!/usr/bin/env python3
"""Check that the CUDA NEGF backend reproduces the CPU result.

Both paths run the same recursion; the GPU one just runs it for many
energies at once, so the two should agree to round-off.  They will not
agree bit for bit -- a batched cuBLAS gemm sums in a different order than
the reference zgemm -- so this compares against a tolerance and prints the
actual deviation it saw rather than only pass/fail.

The backend is selected through the VIDES_GPU environment variable, which
is read once per process, so each case is run twice in fresh subprocesses.

    python3 test/test_gpu_vs_cpu.py              # every case
    python3 test/test_gpu_vs_cpu.py --case gnr   # just one
    python3 test/test_gpu_vs_cpu.py --rtol 1e-9  # stricter

Run it from the directory holding NanoTCAD_ViDESmod.so, or point
PYTHONPATH at it.  On a machine with no GPU both runs land on the CPU and
the test is vacuous -- it says so rather than reporting a pass.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np


# ----------------------------------------------------------------------
# Cases.  Each builds a small device and returns (charge, E, T).
# Keep them small: this is a correctness check, not a benchmark.
# ----------------------------------------------------------------------

def case_cnt():
    """CNT real space -> CNT_charge_T -> VIDES_RGF_STD."""
    from NanoTCAD_ViDES import nanotube
    d = nanotube(10, 2.0)
    d.Elower, d.Eupper, d.dE = -3.0, 3.0, 0.02
    d.Phi = -0.2 * np.ones(d.n * d.Nc)
    d.mu2 = -0.3
    d.charge_T()
    return d.charge, d.E, d.T


def case_cntmode():
    """CNT mode space -> CNTmode_charge_T -> VIDES_RGF_MODE."""
    from NanoTCAD_ViDES import nanotube
    d = nanotube(10, 2.0)
    d.Nmodes = 4
    d.Elower, d.Eupper, d.dE = -3.0, 3.0, 0.02
    d.Phi = -0.2 * np.ones(d.n * d.Nc)
    d.mu2 = -0.3
    d.mode_charge_T()
    return d.charge, d.E, d.T


def case_gnr():
    """Graphene nanoribbon -> GNR_charge_T -> VIDES_RGF_STD."""
    from NanoTCAD_ViDES import nanoribbon
    d = nanoribbon(3, 1)
    d.Elower, d.Eupper, d.dE = -3.0, 3.0, 0.02
    d.Phi = -0.2 * d.z
    d.mu2 = -0.3
    d.charge_T()
    return d.charge, d.E, d.T


def _hamiltonian(biased, offgrid=False):
    from NanoTCAD_ViDES import Hamiltonian
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))
    from GNR import GNR
    h = GNR(5, 6)
    d = Hamiltonian(5, 6)
    d.H = h
    if offgrid:
        # Shift the grid by half a step so no energy lands on E ~ 0.
        d.Elower, d.Eupper, d.dE = -2.99, 2.99, 0.02
    else:
        d.Elower, d.Eupper, d.dE = -3.0, 3.0, 0.02
    d.eta = 1e-5
    if biased:
        d.Phi = -0.2 * np.ones(d.n * d.Nc)
        d.mu2 = -0.3
    d.charge_T()
    return d.charge, d.E, d.T


def case_hamiltonian():
    """Generic tight-binding Hamiltonian -> H_charge_T -> VIDES_RGF_LAKE,
    biased like the CNT and GNR cases.  Same path as Zincblend / nanowire.

    The grid is offset by half a step: with Phi = -0.2 the ribbon's
    zigzag edge states sit at E = +0.2 eV, and an energy point landing
    exactly there (as -3 + 160*0.02 does) makes the inversion nearly
    singular at eta = 1e-5.  Both backends are then round-off dominated
    at that one point, so it says nothing about whether the GPU is right.
    """
    return _hamiltonian(True, offgrid=True)


def case_hamiltonian_ongrid():
    """'hamiltonian' with an energy point exactly on the edge-state
    resonance at E = +0.2 eV.  Diagnostic: on a Colab T4 this disagreed by
    ~7e-7 in charge, all of it from that single ill-conditioned point."""
    return _hamiltonian(True, offgrid=False)


def case_hamiltonian_eq():
    """The same device at equilibrium (Phi = 0, mu1 = mu2 = 0), on the
    unshifted grid, so a point again lands on the edge-state resonance
    (here at E ~ 0).  Diagnostic, as 'hamiltonian_ongrid'."""
    return _hamiltonian(False)


CASES = {
    "cnt": case_cnt,
    "cntmode": case_cntmode,
    "gnr": case_gnr,
    "hamiltonian": case_hamiltonian,
    "hamiltonian_default": case_hamiltonian,
    "hamiltonian_ongrid": case_hamiltonian_ongrid,
    "hamiltonian_eq": case_hamiltonian_eq,
}

# Extra environment per case, applied to both the CPU and the GPU run.
# The Hamiltonian path's contact self-energy is decimation on the GPU but
# the eigen method on the CPU by default (vides_selfh_use_decimation), so
# 'hamiltonian' pins both runs to decimation: any difference is then the
# GPU's.  'hamiltonian_default' keeps the defaults, i.e. what a user gets.
CASE_ENV = {
    "hamiltonian": {"VIDES_SELFH": "dec"},
    "hamiltonian_ongrid": {"VIDES_SELFH": "dec"},
    "hamiltonian_eq": {"VIDES_SELFH": "dec"},
}

# Cases with an energy point on a near-singular resonance by construction;
# reported but not counted as failures.
DIAGNOSTIC_ONLY = {"hamiltonian_ongrid", "hamiltonian_eq"}


# ----------------------------------------------------------------------
# Worker: run one case under the backend the environment selects.
# ----------------------------------------------------------------------

def worker(name, outfile):
    charge, E, T = CASES[name]()
    n = min(len(E), len(T))
    np.savez(outfile,
             charge=np.asarray(charge, dtype=float),
             E=np.asarray(E, dtype=float)[:n],
             T=np.asarray(T, dtype=float)[:n],
             gpu=np.array([1 if os.environ.get("VIDES_GPU", "1") != "0" else 0]))


def run_backend(name, use_gpu, outfile):
    env = dict(os.environ)
    env["VIDES_GPU"] = "1" if use_gpu else "0"
    env.update(CASE_ENV.get(name, {}))
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [os.getcwd(), env.get("PYTHONPATH", "")] if p)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--worker", name, outfile],
        env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise RuntimeError("%s run of case '%s' failed"
                           % ("GPU" if use_gpu else "CPU", name))
    return proc.stdout


def deviation(a, b):
    """Max absolute difference, the same relative to the array's largest
    value, and the worst element-wise relative difference.

    Pass/fail uses the middle one.  Element-wise relative error is
    misleading here: transmission inside a band gap is exponentially small
    (1e-5 .. 1e-10), so round-off that is 1e-9 of the peak transmission
    shows up as a "relative" error of 1e-5 or worse on those points.  It is
    still printed so a genuine problem confined to small values stays
    visible.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return float("inf"), float("inf"), float("inf")
    if a.size == 0:
        return 0.0, 0.0, 0.0
    d = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    peak = float(scale.max())
    nz = scale > 0
    rel = np.zeros_like(d)
    rel[nz] = d[nz] / scale[nz]
    return (float(d.max()),
            float(d.max()) / peak if peak > 0 else 0.0,
            float(rel.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2, metavar=("CASE", "OUT"),
                    help=argparse.SUPPRESS)
    ap.add_argument("--case", choices=sorted(CASES), action="append",
                    help="run only this case (repeatable)")
    ap.add_argument("--rtol", type=float, default=1e-6,
                    help="max tolerated deviation, relative to each array's peak value (default 1e-6)")
    args = ap.parse_args()

    if args.worker:
        worker(args.worker[0], args.worker[1])
        return 0

    names = args.case or sorted(CASES)
    tmp = tempfile.mkdtemp(prefix="vides_gpu_check_")
    failures, skipped = [], []

    for name in names:
        print("=" * 68)
        print("case: %s" % name)
        print("=" * 68)

        cpu_out = os.path.join(tmp, name + "_cpu.npz")
        gpu_out = os.path.join(tmp, name + "_gpu.npz")

        run_backend(name, False, cpu_out)
        banner = run_backend(name, True, gpu_out)

        for line in banner.splitlines():
            if "NEGF backend" in line:
                print("  " + line.strip())
                break

        cpu = np.load(cpu_out)
        gpu = np.load(gpu_out)

        if "no CUDA device" in banner or "GPU disabled" in banner:
            print("  no usable GPU -- both runs used the CPU, nothing compared")
            skipped.append(name)
            continue

        worst = 0.0
        for key in ("charge", "T", "E"):
            adiff, rdiff, erel = deviation(cpu[key], gpu[key])
            worst = max(worst, rdiff)
            status = "ok" if rdiff <= args.rtol else "FAIL"
            print("  %-7s max|d| = %-10.3g  rel to peak = %-10.3g"
                  "  worst element rel = %-10.3g  %s"
                  % (key, adiff, rdiff, erel, status))

        dT = np.abs(cpu["T"] - gpu["T"])
        if dT.size:
            k = int(dT.argmax())
            print("  worst T point: E = %+.6f eV  (T_cpu = %.6g, T_gpu = %.6g)"
                  % (cpu["E"][k], cpu["T"][k], gpu["T"][k]))
        if worst > args.rtol:
            if name in DIAGNOSTIC_ONLY:
                print("  (diagnostic case: not counted as a failure)")
            else:
                failures.append(name)

    print("=" * 68)
    if skipped:
        print("SKIPPED (no GPU): %s" % ", ".join(skipped))
    if failures:
        print("FAILED: %s" % ", ".join(failures))
        return 1
    if not skipped:
        print("all cases agree within rtol=%g" % args.rtol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
