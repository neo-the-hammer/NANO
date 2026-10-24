"""S6 -- per-call GPU overhead: persistent cuBLAS handle and memory pool.

Structures summed over a transverse wave vector (graphene, bilayer, the
graphene FET of R1) call the solver once per k: thousands of short calls
(R1's graphene tutorial made 1617).  The first CUDA version created a
cuBLAS handle and ~30 device buffers in every call.  Now the handle lives
for the whole process and buffers return to a pool for the next call.

Native harness: ncalls consecutive GPU calls of the same small problem,
in two processes, VIDES_GPU_POOL=0 (fresh buffers each call, as before)
and VIDES_GPU_POOL=1 (pool, the default); the handle is persistent in
both, so the difference is the pool alone.  The first call (context and
handle creation) is reported separately.  The CPU time per call (all
threads) is the bar the GPU has to clear for such problems.
"""

from common import Step, run_native, fmt_t, CPU_C, GPU_C, ALT_C

ID = "S6"
TITLE = "Per-call GPU overhead: memory pool, persistent handle"
KIND = "speed"


def cases(quick):
    if quick:
        return [(2, 20, 32), (6, 20, 64), (12, 30, 64)], 30
    return [(1, 160, 128), (2, 40, 32), (6, 40, 64), (12, 40, 128), (24, 40, 128)], 60


def steps(ctx):
    cs, ncalls = cases(ctx["quick"])
    out = []
    for n, Nc, NB in cs:
        for pool in ("0", "1"):
            a = ["calls", "std", n, Nc, NB, ncalls]
            out.append(Step("n=%d Nc=%d NB=%d x%d, pool %s" % (n, Nc, NB, ncalls, "on" if pool == "1" else "off"),
                            "rgf", 10.0 * NB * Nc * n ** 3 * ncalls * 1.1 + 2e8 * ncalls,
                            (lambda a=a, p=pool: run_native(ctx["exe"], a, env={"VIDES_GPU_POOL": p})), True))
    return out


def pairs(recs):
    by = {}
    for r in recs:
        if "error" not in r:
            by.setdefault((r["n"], r["Nc"], r["NB"]), {})["on" if r["pool"] else "off"] = r
    return [(k, v["off"], v["on"]) for k, v in sorted(by.items()) if len(v) == 2]


def analyze(recs, ctx):
    rows, gains, gpu = [], [], False
    for (n, Nc, NB), off, on in pairs(recs):
        if not on["gpu"]:
            rows.append(["n=%d Nc=%d NB=%d" % (n, Nc, NB), "-", "-", "-", "-", fmt_t(on["t_cpu_per_call"]).strip(), "-"])
            continue
        gpu = True
        g = off["t_gpu_per_call"] / on["t_gpu_per_call"]
        gains.append(g)
        rows.append(["n=%d Nc=%d NB=%d" % (n, Nc, NB), fmt_t(on["t_gpu_first"]).strip(),
                     fmt_t(off["t_gpu_per_call"]).strip(), fmt_t(on["t_gpu_per_call"]).strip(), "%.2fx" % g,
                     fmt_t(on["t_cpu_per_call"]).strip(),
                     "%.1fx" % (on["t_cpu_per_call"] / on["t_gpu_per_call"])])
    n_err = sum(1 for r in recs if "error" in r)
    if n_err:
        verdict, head = "FAIL", "%d run(s) errored" % n_err
    elif not gpu:
        verdict, head = "NO GPU", "CPU timings only"
    else:
        verdict, head = "INFO", "pool cuts the GPU time per call by up to %.2fx" % max(gains)
    return {"verdict": verdict, "headline": head,
            "table": {"cols": ["problem", "GPU first call", "per call, pool off", "per call, pool on",
                               "pool gain", "CPU per call", "GPU vs CPU (pool on)"], "rows": rows}}


def plot(recs, summ, ctx, plt):
    P = [p for p in pairs(recs) if p[2]["gpu"]]
    if not P:
        return []
    fig, a = plt.subplots(figsize=(9, 3.8))
    fig.suptitle("S6  Time per call of a small batched solve (many calls in a row)", x=0.01, ha="left")
    x = list(range(len(P)))
    w = 0.27
    a.bar([i - w for i in x], [1e3 * p[1]["t_gpu_per_call"] for p in P], w, color=ALT_C[3], label="GPU, pool off")
    a.bar(x, [1e3 * p[2]["t_gpu_per_call"] for p in P], w, color=GPU_C, label="GPU, pool on (default)")
    a.bar([i + w for i in x], [1e3 * p[2]["t_cpu_per_call"] for p in P], w, color=CPU_C, label="CPU, all threads")
    a.set_xticks(x)
    a.set_xticklabels(["n=%d Nc=%d\nNB=%d" % k for k, _, _ in P], fontsize=8)
    a.set_ylabel("ms per call")
    a.set_yscale("log")
    a.legend(fontsize=7)
    fig.tight_layout()
    return [fig]
