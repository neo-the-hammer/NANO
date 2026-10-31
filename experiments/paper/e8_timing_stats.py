#!/usr/bin/env python3
"""E8 -- timing statistics: how certain the speedups are.

A single timing hides noise from turbo clocks, page faults, the GPU's
power state and other tenants of a shared cloud machine.  A paper needs
repeated measurements and an interval.  For three devices, one per
transport path, each backend runs in 3 independent processes with 4
solves each; the first solve of a GPU process (start-up) is dropped.
Reported: mean and standard deviation of the time per solve, the
coefficient of variation, and the speedup with a 95% bootstrap
confidence interval (2000 resamples of the CPU and GPU times).
"""

import random

import _runner  # noqa: F401
from common import Step, run_device, u_device, fmt_t, CPU_C, GPU_C

ID = "E8"
TITLE = "Timing statistics: repeated runs and 95% confidence intervals of the speedups"
KIND = "speed"
PROCS, REPEATS = 3, 4


def devices(quick):
    g = [-1.5025, 1.5025, 0.01] if quick else [-1.5025, 1.5025, 0.005]
    if quick:
        return [("cnt 13x5", {"kind": "cnt", "size": [13, 5.0], "grid": g}),
                ("gnr 8x5", {"kind": "gnr", "size": [8, 5.0], "grid": g})]
    return [("cnt 19x10", {"kind": "cnt", "size": [19, 10.0], "grid": g}),
            ("gnr 12x10", {"kind": "gnr", "size": [12, 10.0], "grid": g}),
            ("H 16x40", {"kind": "hamiltonian", "size": [16, 40], "grid": g, "eta": 1e-5})]


def steps(ctx):
    out = []
    for name, spec in devices(ctx["quick"]):
        for gpu in (False, True):
            for p in range(PROCS):
                out.append(Step("%s %s, process %d" % (name, "GPU" if gpu else "CPU", p + 1),
                                "dev_%s_%s" % (spec["kind"], "gpu" if gpu else "cpu"),
                                u_device(spec, gpu, None, REPEATS),
                                (lambda spec=spec, gpu=gpu, name=name, p=p:
                                 dict(run_device(spec, gpu, repeat=REPEATS), device=name, gpu_req=gpu, proc=p)),
                                gpu))
    return out


def samples(recs, name, gpu):
    out = []
    for r in recs:
        if "error" not in r and r.get("device") == name and r["gpu_req"] == gpu:
            out.extend(r["times"][1:] if gpu else r["times"])
    return out


def mean_sd(v):
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
    return m, sd


def boot_ci(c, g, n=2000, seed=1):
    rnd = random.Random(seed)
    rs = []
    for _ in range(n):
        mc = sum(rnd.choice(c) for _ in c) / len(c)
        mg = sum(rnd.choice(g) for _ in g) / len(g)
        rs.append(mc / mg)
    rs.sort()
    return rs[int(0.025 * n)], rs[int(0.975 * n) - 1]


def analyze(recs, ctx):
    rows, heads, gpu = [], [], False
    for name, _ in devices(ctx["quick"]):
        c, g = samples(recs, name, False), samples(recs, name, True)
        gpu = gpu or any(r.get("gpu_used") for r in recs if r.get("device") == name and r.get("gpu_req"))
        if len(c) < 2 or len(g) < 2:
            continue
        mc, sc = mean_sd(c)
        mg, sg = mean_sd(g)
        lo, hi = boot_ci(c, g)
        rows.append([name, len(c), "%s +- %s" % (fmt_t(mc).strip(), fmt_t(sc).strip()), "%.1f%%" % (100 * sc / mc),
                     len(g), "%s +- %s" % (fmt_t(mg).strip(), fmt_t(sg).strip()), "%.1f%%" % (100 * sg / mg),
                     "%.2fx" % (mc / mg), "[%.2f, %.2f]" % (lo, hi)])
        heads.append("%s %.1fx [%.1f, %.1f]" % (name, mc / mg, lo, hi))
    n_err = sum(1 for r in recs if "error" in r)
    if n_err:
        verdict, head = "FAIL", "%d run(s) errored" % n_err
    elif not gpu:
        verdict, head = "NO GPU", "CPU only"
    else:
        verdict, head = "INFO", "speedup with 95% CI: " + "; ".join(heads)
    return {"verdict": verdict, "headline": head,
            "table": {"cols": ["device", "CPU n", "CPU time (mean +- sd)", "CPU CV", "GPU n",
                               "GPU time (mean +- sd)", "GPU CV", "speedup", "95% CI"], "rows": rows}}


def plot(recs, summ, ctx, plt):
    names = [n for n, _ in devices(ctx["quick"])]
    data = [(n, samples(recs, n, False), samples(recs, n, True)) for n in names]
    data = [d for d in data if d[1] and d[2]]
    if not data:
        return []
    fig, a = plt.subplots(figsize=(8, 3.6))
    fig.suptitle("E8  Spread of the time per solve (every repeat of every process)", x=0.01, ha="left")
    pos, vals, cols = [], [], []
    for i, (n, c, g) in enumerate(data):
        pos += [3 * i, 3 * i + 1]
        vals += [c, g]
        cols += [CPU_C, GPU_C]
    bp = a.boxplot(vals, positions=pos, widths=0.7, patch_artist=True)
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    a.set_xticks([3 * i + 0.5 for i in range(len(data))])
    a.set_xticklabels([d[0] for d in data], fontsize=8)
    a.set_yscale("log")
    a.set_ylabel("seconds per solve (blue CPU, orange GPU)")
    fig.tight_layout()
    return [fig]


if __name__ == "__main__":
    _runner.run_one(__import__(__name__))
