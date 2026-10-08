"""S1 -- speed of the NEGF solver (RGF), CPU vs GPU, along each axis.

The batched solver alone, through the native harness, swept along
  block size n           (cost ~ n^3: where a GPU pays off)
  number of blocks Nc    (cost ~ Nc: sequential in the recursion)
  energies per batch NB  (the axis the GPU parallelises over)
GPU time is steady state (best of two calls after a warm-up call).  The
numerical agreement of every timed run is checked too (as in C1).
"""

from common import u_rgf, Step, run_native, fmt_e, fmt_t, CPU_C, GPU_C, MUTED

ID = "S1"
TITLE = "NEGF solver speed: scaling in n, Nc, NB"
KIND = "speed"


def sweeps(quick):
    if quick:
        return {"n": ([8, 16, 32], 20, 128), "Nc": (16, [10, 40], 128),
                "NB": (16, 20, [32, 128, 512])}
    return {"n": ([8, 16, 24, 32, 48], 40, 256), "Nc": (24, [20, 40, 80, 160], 256),
            "NB": (24, 40, [32, 64, 128, 256, 512, 1024])}


def steps(ctx):
    out = []
    for axis, (n, Nc, NB) in sweeps(ctx["quick"]).items():
        vals = {"n": n, "Nc": Nc, "NB": NB}[axis]
        for v in vals:
            p = {"n": n, "Nc": Nc, "NB": NB}
            p[axis] = v
            a = ["rgf", "std", p["n"], p["Nc"], p["NB"], 1e-5, 2]
            units = u_rgf(p["n"], p["Nc"], p["NB"]) * (1 + (3 * 0.5 if ctx["gpu"] else 0))

            def fn(a=a, axis=axis):
                r = run_native(ctx["exe"], a)
                r["axis"] = axis
                return r
            out.append(Step("%s sweep: n=%d Nc=%d NB=%d" % (axis, p["n"], p["Nc"], p["NB"]),
                            "rgf", units, fn, ctx["gpu"]))
    return out


def analyze(recs, ctx):
    ok = [r for r in recs if "error" not in r]
    gpu = any(r.get("gpu") for r in ok)
    rows = []
    best = None
    for r in ok:
        sp = r["t_cpu"] / r["t_gpu"] if r.get("gpu") and r["t_gpu"] else None
        if sp and (best is None or sp > best[0]):
            best = (sp, r)
        rows.append([r["axis"], r["n"], r["Nc"], r["NB"], fmt_t(r["t_cpu"]).strip(),
                     fmt_t(r["t_gpu_first"]).strip() if r.get("gpu") else "-",
                     fmt_t(r["t_gpu"]).strip() if r.get("gpu") else "-",
                     "%.1fx" % sp if sp else "-",
                     fmt_e(max(r["A1"]["relpeak"], r["A2"]["relpeak"], r["T"]["relpeak"]))
                     if r.get("gpu") else "-"])
    if len(ok) < len(recs):
        verdict, head = "FAIL", "%d run(s) errored" % (len(recs) - len(ok))
    elif not gpu:
        verdict, head = "NO GPU", "CPU timings only"
    else:
        verdict = "INFO"
        head = "best speedup %.1fx at n=%d Nc=%d NB=%d" % (best[0], best[1]["n"], best[1]["Nc"],
                                                          best[1]["NB"])
    return {"verdict": verdict, "headline": head,
            "table": {"cols": ["sweep", "n", "Nc", "NB", "CPU", "GPU first", "GPU steady",
                               "speedup", "max diff"], "rows": rows}}


def plot(recs, summ, ctx, plt):
    ok = [r for r in recs if "error" not in r]
    axes = ["n", "Nc", "NB"]
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), sharex="col",
                           gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("S1  NEGF solver: time per batched solve and speedup", x=0.01, ha="left")
    for k, axis in enumerate(axes):
        rs = sorted([r for r in ok if r["axis"] == axis], key=lambda r: r[axis])
        if not rs:
            continue
        x = [r[axis] for r in rs]
        t, s = ax[0][k], ax[1][k]
        t.loglog(x, [r["t_cpu"] for r in rs], "o-", color=CPU_C, label="CPU")
        g = [r for r in rs if r.get("gpu")]
        if g:
            t.loglog([r[axis] for r in g], [r["t_gpu"] for r in g], "s-", color=GPU_C,
                     label="GPU (steady)")
            t.loglog([r[axis] for r in g], [r["t_gpu_first"] for r in g], "s:", color=GPU_C,
                     alpha=0.5, label="GPU (first call)")
            s.semilogx([r[axis] for r in g], [r["t_cpu"] / r["t_gpu"] for r in g], "D-",
                       color=MUTED)
            s.axhline(1, color="#d03b3b", lw=1, ls="--")
        fixed = ", ".join("%s=%d" % (a, rs[0][a]) for a in axes if a != axis)
        t.set_title("vary %s  (%s)" % (axis, fixed))
        t.set_ylabel("seconds")
        t.legend(fontsize=7)
        s.set_xlabel(axis)
        s.set_ylabel("speedup CPU/GPU")
    fig.tight_layout()
    return [fig]
