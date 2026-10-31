#!/usr/bin/env python3
"""E7 -- the contact self-energy policy of the Hamiltonian path.

The Hamiltonian path (generic tight-binding devices, nanowires, xyz
structures) needs a contact self-energy per energy.  Two methods exist:
the eigen method (selfH_new, zgeev, CPU only, accurate to ~1e-12) and
Sancho-Rubio decimation (selfH_dec, batched on the GPU).  On the T4 the
self-energy took 83-97% of a GPU solve, so its placement decides the
speed.  This experiment backs the default (the eigen method on the CPU
threads, pipelined with the GPU solve) with measurements:

  CPU, eigen, all threads              the best CPU-only configuration
  GPU solve + eigen on 1 CPU thread    pipeline, one core
  GPU solve + eigen on all threads     the default
  GPU solve + GPU decimation           VIDES_SELFH=dec

for ribbons of width 8, 16 and 24 (40 slices), a bilayer flake summed over
32 ky (small blocks, many short solves) and a silicon nanowire (10
orbitals per atom), at eta = 1e-5.  Accuracy against the CPU eigen
run is reported too (decimation is the less accurate method, A1).
"""

import _runner  # noqa: F401
from common import Step, run_device, steady, rel_to_peak, u_device, fmt_t, fmt_e, ALT_C, CPU_C, GPU_C

ID = "E7"
TITLE = "Contact self-energy policy on the Hamiltonian path: eigen pipelined vs GPU decimation"
KIND = "speed"

CONFIGS = [("CPU eigen, all threads", False, "eig", {}),
           ("GPU + eigen, 1 thread", True, "eig", {"VIDES_THREADS": "1"}),
           ("GPU + eigen, all threads (default)", True, "eig", {}),
           ("GPU + GPU decimation", True, "dec", {})]


def devices(quick):
    g = [-1.5025, 1.5025, 0.01] if quick else [-1.5025, 1.5025, 0.005]
    d = [("H 8x40", {"kind": "hamiltonian", "size": [8, 40], "grid": g, "eta": 1e-5}),
         ("H 16x40", {"kind": "hamiltonian", "size": [16, 40], "grid": g, "eta": 1e-5}),
         # small blocks, one short solve per ky: on Kaggle the eigen default made R2's
         # bilayer test 0.5x the CPU, where GPU decimation had been faster
         ("bilayer 2D 15nm x32 ky", {"kind": "bilayer", "size": [15.0, 32],
                                     "grid": [-0.4975, 0.4975, 0.005 if not quick else 0.01], "eta": 1e-5})]
    if not quick:
        d += [("H 24x40", {"kind": "hamiltonian", "size": [24, 40], "grid": g, "eta": 1e-5}),
              ("Si nanowire 3.85x15A", {"kind": "snw", "size": [3.85, 15.0], "grid": [0.05, 9.95, 0.05],
                                        "eta": 1e-5})]
    return d


def steps(ctx):
    out = []
    for name, spec in devices(ctx["quick"]):
        for label, gpu, selfh, env in CONFIGS:
            out.append(Step("%s: %s" % (name, label), "dev_%s_%s" % (spec["kind"], "gpu" if gpu else "cpu"),
                            u_device(spec, gpu, selfh, 2 if gpu else 1),
                            (lambda spec=spec, gpu=gpu, selfh=selfh, env=env, name=name, label=label:
                             dict(run_device(spec, gpu, selfh=selfh, repeat=2 if gpu else 1, env=env),
                                  device=name, config=label)), gpu))
    return out


def analyze(recs, ctx):
    by = {}
    for r in recs:
        if "error" not in r and "device" in r:
            by.setdefault(r["device"], {})[r["config"]] = r
    rows, wins, gpu = [], [], False
    for name, m in by.items():
        ref = m.get(CONFIGS[0][0])
        t = {k: steady(v["times"]) for k, v in m.items()}
        best = min(t, key=t.get)
        for label, *_ in CONFIGS:
            if label not in m:
                continue
            r = m[label]
            gpu = gpu or r.get("gpu_used", False)
            d = max(rel_to_peak(ref["T"], r["T"]),
                    abs(ref["current"] - r["current"]) / max(abs(ref["current"]), 1e-300)) if ref and r is not ref else 0.0
            rows.append([name, label, fmt_t(t[label]).strip(),
                         "%.2fx" % (t[CONFIGS[0][0]] / t[label]) if CONFIGS[0][0] in t else "-",
                         fmt_e(d) if r is not ref else "reference", "fastest" if label == best else ""])
        dflt, dec = CONFIGS[2][0], CONFIGS[3][0]
        if dflt in t and dec in t:
            wins.append((t[dec] / t[dflt], name))
    n_err = sum(1 for r in recs if "error" in r)
    if n_err:
        verdict, head = "FAIL", "%d run(s) errored" % n_err
    elif not gpu:
        verdict, head = "NO GPU", "CPU only"
    else:
        verdict = "INFO"
        better = sum(1 for w, _ in wins if w >= 0.95)
        head = ("default (eigen pipelined, all threads) vs GPU decimation: %s; default no slower in %d/%d devices"
                % (", ".join("%.2fx %s" % w for w in wins), better, len(wins)))
    return {"verdict": verdict, "headline": head,
            "table": {"cols": ["device", "configuration", "time per solve", "speedup vs CPU",
                               "diff vs CPU eigen", ""], "rows": rows}}


def plot(recs, summ, ctx, plt):
    by = {}
    for r in recs:
        if "error" not in r and "device" in r:
            by.setdefault(r["device"], {})[r["config"]] = steady(r["times"])
    if not by:
        return []
    fig, a = plt.subplots(figsize=(9, 3.8))
    fig.suptitle("E7  Hamiltonian path: where the contact self-energy is computed", x=0.01, ha="left")
    names = list(by)
    w = 0.2
    cols = [CPU_C, ALT_C[2], GPU_C, ALT_C[3]]
    for k, (label, *_) in enumerate(CONFIGS):
        xs = [i + (k - 1.5) * w for i, n in enumerate(names) if label in by[n]]
        ys = [by[n][label] for n in names if label in by[n]]
        a.bar(xs, ys, w, color=cols[k], label=label)
    a.set_xticks(range(len(names)))
    a.set_xticklabels(names, fontsize=8)
    a.set_yscale("log")
    a.set_ylabel("seconds per solve")
    a.legend(fontsize=7)
    fig.tight_layout()
    return [fig]


if __name__ == "__main__":
    _runner.run_one(__import__(__name__))
