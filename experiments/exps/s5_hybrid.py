"""S5 -- CPU+GPU co-execution: one batch split across both, concurrently.

For each problem size the same NB energies are solved four ways:

  CPU 1 thread      the stock path
  CPU all threads   S4
  GPU only          batched cuBLAS, the CPU idle
  hybrid            the proposed scheduler: the GPU takes part of each
                    batch on a background thread while the CPU threads
                    take the rest; the split follows the throughput each
                    side measured on earlier batches of this shape, so
                    both finish together

Swept over the block size n of the solver (where the balance moves from
"CPU faster" at small n to "GPU much faster" at large n) and over the cell
size of the decimation.  The hybrid time is the best after the first call
(the first call only probes the split).  Ideal co-execution of two
independent workers would take 1 / (1/t_cpu + 1/t_gpu): shown as
"ideal", with hybrid efficiency = ideal / hybrid.
"""

from common import Step, run_native, with_gpu, u_rgf, u_decim, fmt_t, fmt_e, CPU_C, GPU_C, ALT_C, MUTED

ID = "S5"
TITLE = "CPU+GPU co-execution (hybrid split) vs either alone"
KIND = "speed"


def plan(quick):
    if quick:
        return [4, 8, 16, 24], 30, 128, [4, 8, 12], 64
    return [4, 6, 8, 12, 16, 24, 32, 48], 40, 256, [4, 6, 8, 12, 16], 128


def steps(ctx):
    ns, Nc, NB, dn, dNB = plan(ctx["quick"])
    g, out = ctx["gpu"], []
    for n in ns:
        a = ["modes", "std", n, Nc, NB, 1e-5, 2]
        out.append(Step("solver n=%d Nc=%d NB=%d" % (n, Nc, NB), "rgf", with_gpu(u_rgf(n, Nc, NB), 8, g) * 1.6,
                        (lambda a=a: dict(run_native(ctx["exe"], a), part="rgf")), g))
    for n in dn:
        a = ["dmodes", "hsrc", n, dNB, 1e-5, 2]
        out.append(Step("decimation n=%d NB=%d" % (n, dNB), "decim", with_gpu(u_decim(n, dNB), 8, g) * 1.6,
                        (lambda a=a: dict(run_native(ctx["exe"], a), part="decim")), g))
    return out


def ideal(tc, tg):
    return 1.0 / (1.0 / tc + 1.0 / tg) if tc and tg else float("nan")


def analyze(recs, ctx):
    rows, gains, gpu = [], [], False
    for r in recs:
        if "error" in r:
            continue
        name = ("solver" if r["part"] == "rgf" else "decimation") + " n=%d" % r["n"]
        if not r["gpu"]:
            rows.append([name, fmt_t(r["t_cpu1"]).strip(), fmt_t(r["t_cpuN"]).strip(), "-", "-", "-", "-", "-", "-"])
            continue
        gpu = True
        tc, tg, th = r["t_cpuN"], r["t_gpu"], r["t_hybrid"]
        best1 = min(tc, tg)
        gains.append((best1 / th, name))
        diff = (max(r["hyb_A1"]["relpeak"], r["hyb_T"]["relpeak"]) if r["part"] == "rgf" else r["hyb_rel_diff"])
        rows.append([name, fmt_t(r["t_cpu1"]).strip(), fmt_t(tc).strip(), fmt_t(tg).strip(), fmt_t(th).strip(),
                     "%.0f%%" % (100 * r["cpu_share"]) if r["cpu_share"] >= 0 else "-",
                     "%.2fx" % (best1 / th), "%.0f%%" % (100 * ideal(tc, tg) / th), fmt_e(diff)])
    n_err = sum(1 for r in recs if "error" in r)
    if n_err:
        verdict, head = "FAIL", "%d run(s) errored" % n_err
    elif not gpu:
        verdict, head = "NO GPU", "CPU timings only"
    else:
        g = max(gains)
        verdict = "INFO"
        head = ("hybrid vs the faster of CPU/GPU alone: best %.2fx (%s), median %.2fx"
                % (g[0], g[1], sorted(x for x, _ in gains)[len(gains) // 2]))
    return {"verdict": verdict, "headline": head,
            "table": {"cols": ["problem", "CPU 1 thr", "CPU all thr", "GPU only", "hybrid", "CPU share",
                               "hybrid vs best single", "vs ideal", "hybrid diff"], "rows": rows}}


def plot(recs, summ, ctx, plt):
    ok = [r for r in recs if "error" not in r and r["gpu"]]
    if not ok:
        return []
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex="col", gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("S5  Co-execution: one batch split between the GPU and the CPU threads", x=0.01, ha="left")
    for k, (part, title) in enumerate((("rgf", "NEGF solver"), ("decim", "decimation self-energy"))):
        rs = sorted([r for r in ok if r["part"] == part], key=lambda r: r["n"])
        if not rs:
            continue
        x = [r["n"] for r in rs]
        a, b = ax[0][k], ax[1][k]
        a.loglog(x, [r["t_cpu1"] for r in rs], "o:", color=CPU_C, alpha=0.6, label="CPU, 1 thread (stock)")
        a.loglog(x, [r["t_cpuN"] for r in rs], "o-", color=CPU_C, label="CPU, all threads")
        a.loglog(x, [r["t_gpu"] for r in rs], "s-", color=ALT_C[3], label="GPU only")
        a.loglog(x, [r["t_hybrid"] for r in rs], "D-", color=GPU_C, lw=2, label="hybrid (proposed)")
        a.loglog(x, [ideal(r["t_cpuN"], r["t_gpu"]) for r in rs], "--", color=MUTED, label="ideal co-execution")
        a.set_title(title)
        a.set_ylabel("seconds")
        a.legend(fontsize=7)
        b.plot(x, [100 * r["cpu_share"] for r in rs], "D-", color=GPU_C)
        b.set_ylabel("CPU share (%)")
        b.set_xlabel("block size n" if part == "rgf" else "lead slice n (cell 4n)")
        b.set_xscale("log")
    fig.tight_layout()
    return [fig]
