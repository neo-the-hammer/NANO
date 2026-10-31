#!/usr/bin/env python3
"""E6 -- several GPUs: a 2D device's transverse wave vectors split across them.

2D structures (graphene, bilayer, TMD FETs) cost one NEGF solve per
transverse wave vector ky, and the ky are independent: the natural unit
to spread over GPUs, with no communication but the final sum.  A
graphene flake (20 nm, 64 ky; quick 10 nm, 16 ky) is solved

  1 process on GPU 0 (the reference)
  G processes, one per GPU (CUDA_VISIBLE_DEVICES), when G > 1 GPUs exist
  2 processes sharing GPU 0 (does one GPU gain from concurrent work?)

each process taking an interleaved share of the ky.  The parts must add up
to the single-process T(E) and charge to round-off (1e-10 relative to
peak), and the wall time of each layout gives the parallel speedup and
efficiency.  Kaggle's "T4 x2" runtime is the cheapest place to see G = 2.
"""

import subprocess
import threading
import time

import _runner  # noqa: F401
import _proc
from common import Step, fmt_t, fmt_e, rel_to_peak, GPU_C, ALT_C

ID = "E6"
TITLE = "Multi-GPU: transverse wave vectors of a 2D device split across GPUs"
KIND = "speed"
TOL = 1e-10


def gpu_count():
    try:
        p = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return len([l for l in p.stdout.splitlines() if l.startswith("GPU ")]) if p.returncode == 0 else 0
    except Exception:
        return 0


def plan(quick):
    import math
    L, nk = (10.0, 16) if quick else (20.0, 64)
    kF = 2 * math.pi / (3 * math.sqrt(3) * 0.144)
    lo, hi = kF - 1.0, kF + 1.0
    dk = (hi - lo) / nk
    ks = [lo + i * dk for i in range(nk + 1)]          # as graphene.charge_T: both ends
    return {"L": L, "dk": dk, "grid": [-0.4975, 0.4975, 0.005], "phi": -0.1, "mu2": -0.2}, ks


def layout(n_proc, devices, quick):
    """Run n_proc workers at once, worker i on devices[i] (None: inherit)."""
    base, ks = plan(quick)
    parts = [ks[i::n_proc] for i in range(n_proc)]
    res = [None] * n_proc
    errs = []

    def go(i):
        env = {} if devices[i] is None else {"CUDA_VISIBLE_DEVICES": str(devices[i])}
        try:
            res[i] = _proc.run_worker("kslice_worker.py", dict(base, k=parts[i]), True, env=env)
        except Exception as e:  # reported below
            errs.append(str(e))

    t0 = time.time()
    th = [threading.Thread(target=go, args=(i,)) for i in range(n_proc)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.time() - t0
    if errs:
        raise RuntimeError(errs[0])
    T = [sum(v) for v in zip(*[r["T"] for r in res])]
    ch = [sum(v) for v in zip(*[r["charge"] for r in res])]
    return {"n_proc": n_proc, "devices": devices, "wall": wall, "T": T, "charge": ch,
            "gpu_used": all(r.get("gpu_used") for r in res), "nk": len(ks),
            "worker_walls": [r["wall"] for r in res]}


def steps(ctx):
    q = ctx["quick"]
    G = gpu_count()
    nk = len(plan(q)[1])
    per = (4e9 if q else 2.5e10)               # nominal CPU-seconds x 1e9 for the whole sweep
    out = [Step("1 process, GPU 0 (%d ky)" % nk, "sim", per,
                (lambda: dict(layout(1, [0 if G else None], q), name="1 GPU")), True)]
    if G > 1:
        out.append(Step("%d processes on %d GPUs" % (G, G), "sim", per / G,
                        (lambda: dict(layout(G, list(range(G)), q), name="%d GPUs" % G)), True))
    out.append(Step("2 processes sharing GPU 0", "sim", per,
                    (lambda: dict(layout(2, [0 if G else None] * 2, q), name="2 processes, 1 GPU")), True))
    return out


def analyze(recs, ctx):
    ok = [r for r in recs if "error" not in r and "wall" in r]
    ref = next((r for r in ok if r["n_proc"] == 1), None)
    n_err = sum(1 for r in recs if "error" in r)
    if n_err and not ref:
        return {"verdict": "FAIL", "headline": "%d run(s) errored" % n_err, "table": None}
    rows, worst, best = [], 0.0, None
    for r in ok:
        dT = rel_to_peak(ref["T"], r["T"]) if r is not ref else 0.0
        dc = rel_to_peak(ref["charge"], r["charge"]) if r is not ref else 0.0
        d = max(dT, dc)
        worst = max(worst, d)
        sp = ref["wall"] / r["wall"]
        n_gpu = len(set(x for x in r["devices"] if x is not None)) or 1
        eff = sp / n_gpu if r is not ref else 1.0
        if r is not ref and n_gpu > 1:
            best = (sp, n_gpu, eff)
        rows.append([r["name"], r["n_proc"], n_gpu, fmt_t(r["wall"]).strip(), "%.2fx" % sp,
                     "%.0f%%" % (100 * eff), fmt_e(d) if r is not ref else "reference"])
    G = gpu_count()
    gpu = any(r.get("gpu_used") for r in ok)
    if n_err:
        verdict, head = "FAIL", "%d run(s) errored" % n_err
    elif not gpu:
        verdict, head = "NO GPU", "no GPU: layouts ran on the CPU"
    else:
        verdict = "PASS" if worst < TOL else ("WARN" if worst < 100 * TOL else "FAIL")
        head = ("split ky sums equal the single run (max diff %s); %s"
                % (fmt_e(worst), ("%d GPUs: %.2fx, %.0f%% efficiency" % (best[1], best[0], 100 * best[2]))
                   if best else "only %d GPU visible: multi-GPU speedup not measurable here" % G))
    return {"verdict": verdict, "headline": head,
            "table": {"cols": ["layout", "processes", "GPUs", "wall time", "speedup", "efficiency",
                               "diff vs 1 process"], "rows": rows}}


def plot(recs, summ, ctx, plt):
    ok = [r for r in recs if "error" not in r and "wall" in r]
    if not ok:
        return []
    fig, a = plt.subplots(figsize=(6, 3.4))
    fig.suptitle("E6  Wall time of one 2D-device solve, by layout", x=0.01, ha="left")
    a.barh([r["name"] for r in ok], [r["wall"] for r in ok],
           color=[GPU_C if r["n_proc"] == 1 else ALT_C[2] for r in ok])
    a.set_xlabel("seconds")
    a.invert_yaxis()
    fig.tight_layout()
    return [fig]


if __name__ == "__main__":
    _runner.run_one(__import__(__name__))
