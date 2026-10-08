"""C1 -- NEGF solver (recursive Green's function), CPU vs GPU.

The batched solver is called directly through the native harness with
identical random Hermitian chains and contact self-energies: the CPU
reference (stock rgfblock / rgfblock_Lake / LDOSMODE per energy) and the
cuBLAS batched kernel.  All three recursion variants, two block sizes, and
one stress case with a tiny eta (nearly singular blocks).

Compared: source LDOS A1, drain LDOS A2 and transmission T, as max|diff|
relative to the array's peak.  Pass: < 1e-8 (round-off is ~1e-13).
"""

from common import u_rgf, Step, run_native, fmt_e, positive, CPU_C, GPU_C, ALT_C, MUTED

ID = "C1"
TITLE = "NEGF solver (RGF): CPU vs GPU, all variants"
KIND = "correctness"
TOL, WARN = 1e-8, 1e-6


def cases(quick):
    if quick:
        return [("std", 8, 20, 64, 1e-5), ("lake", 8, 20, 64, 1e-5),
                ("mode", 8, 20, 64, 1e-5), ("std", 16, 20, 64, 1e-9)]
    return [("std", 8, 30, 128, 1e-5), ("std", 24, 40, 128, 1e-5),
            ("lake", 8, 30, 128, 1e-5), ("lake", 24, 40, 128, 1e-5),
            ("mode", 8, 30, 128, 1e-5), ("mode", 24, 40, 128, 1e-5),
            ("std", 16, 30, 128, 1e-9), ("lake", 16, 30, 128, 1e-9)]


def steps(ctx):
    out = []
    for var, n, Nc, NB, eta in cases(ctx["quick"]):
        args = ["rgf", var, n, Nc, NB, eta, 1]
        units = u_rgf(n, Nc, NB) * (1 + (2 * 0.5 if ctx["gpu"] else 0))
        out.append(Step("%s n=%d Nc=%d NB=%d eta=%g" % (var, n, Nc, NB, eta), "rgf", units,
                        (lambda a=args: run_native(ctx["exe"], a)), ctx["gpu"]))
    return out


def label(r):
    return "%s n=%d Nc=%d eta=%g" % (r["variant"], r["n"], r["Nc"], r["eta"])


def analyze(recs, ctx):
    rows, worst = [], 0.0
    ok = [r for r in recs if "error" not in r]
    gpu = any(r.get("gpu") for r in ok)
    for r in ok:
        if r.get("gpu"):
            w = max(r["A1"]["relpeak"], r["A2"]["relpeak"], r["T"]["relpeak"])
            worst = max(worst, w)
            rows.append([label(r), fmt_e(r["A1"]["relpeak"]), fmt_e(r["A2"]["relpeak"]),
                         fmt_e(r["T"]["relpeak"]), "%.2f s" % r["t_cpu"], "%.3f s" % r["t_gpu"],
                         "PASS" if w < TOL else ("WARN" if w < WARN else "FAIL")])
        else:
            rows.append([label(r), "-", "-", "-", "%.2f s" % r["t_cpu"], "-", "NO GPU"])
    for r in recs:
        if "error" in r:
            rows.append([r.get("label", "?"), "-", "-", "-", "-", "-", "ERROR"])
    if not gpu:
        verdict, head = "NO GPU", "no CUDA device: only the CPU reference ran"
    elif len(ok) < len(recs):
        verdict, head = "FAIL", "%d case(s) errored" % (len(recs) - len(ok))
    else:
        verdict = "PASS" if worst < TOL else ("WARN" if worst < WARN else "FAIL")
        head = "worst rel. diff (A1, A2, T) = %s over %d cases" % (fmt_e(worst), len(ok))
    return {"verdict": verdict, "headline": head, "worst": worst,
            "table": {"cols": ["case", "A1 diff", "A2 diff", "T diff", "CPU", "GPU", "verdict"],
                      "rows": rows}}


def plot(recs, summ, ctx, plt):
    ok = [r for r in recs if "error" not in r]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle("C1  NEGF solver: CPU vs GPU on identical inputs", x=0.01, ha="left")
    # (a) one transmission spectrum per variant, CPU line, GPU dots
    a = ax[0][0]
    seen = {}
    for r in ok:
        if r["variant"] not in seen or r["n"] > seen[r["variant"]]["n"]:
            seen[r["variant"]] = r
    off = 0.0
    for i, (v, r) in enumerate(sorted(seen.items())):
        T = [t + off for t in r["T_cpu"]]
        a.plot(r["E"], T, color=CPU_C, lw=1.4, label="CPU" if i == 0 else None)
        a.plot(r["E"], [t + off for t in r["T_gpu"]], "o", ms=2.4, color=GPU_C,
               label="GPU" if i == 0 else None)
        a.text(r["E"][-1], T[-1], "  " + label(r), fontsize=7, color=MUTED, va="center")
        off += 1.1 * max(r["T_cpu"] + [1e-30])
    a.set_xlabel("E (eV)")
    a.set_ylabel("T(E), offset per variant")
    a.set_title("(a) transmission: CPU line, GPU markers")
    a.legend(loc="upper left")
    # (b) |dT| vs E for every case
    b = ax[0][1]
    for i, r in enumerate(ok):
        if not r.get("gpu"):
            continue
        d = [abs(x - y) for x, y in zip(r["T_cpu"], r["T_gpu"])]
        b.semilogy(r["E"], positive(d), lw=1, color=ALT_C[i % len(ALT_C)], label=label(r))
    b.set_xlabel("E (eV)")
    b.set_ylabel("|T_cpu - T_gpu|")
    b.set_title("(b) transmission difference per energy")
    b.legend(fontsize=6, ncol=2)
    # (c) relpeak per quantity per case
    c = ax[1][0]
    g = [r for r in ok if r.get("gpu")]
    w = 0.27
    for k, (q, col) in enumerate([("A1", CPU_C), ("A2", GPU_C), ("T", "#3a9d5d")]):
        c.bar([i + (k - 1) * w for i in range(len(g))], positive([r[q]["relpeak"] for r in g]),
              w, color=col, label=q)
    c.axhline(TOL, color="#d03b3b", lw=1, ls="--", label="pass threshold %g" % TOL)
    c.set_yscale("log")
    c.set_xticks(range(len(g)))
    c.set_xticklabels([label(r) for r in g], rotation=35, ha="right", fontsize=7)
    c.set_ylabel("max |diff| / peak")
    c.set_title("(c) agreement per quantity")
    c.legend(fontsize=7)
    # (d) worst element-wise relative difference (gap regions)
    d = ax[1][1]
    d.bar(range(len(g)), positive([max(r[q]["worstelem"] for q in ("A1", "A2", "T")) for r in g]),
          color=MUTED)
    d.set_yscale("log")
    d.set_xticks(range(len(g)))
    d.set_xticklabels([label(r) for r in g], rotation=35, ha="right", fontsize=7)
    d.set_ylabel("worst element-wise rel. diff")
    d.set_title("(d) worst single element (dominated by tiny values in gaps)")
    fig.tight_layout()
    return [fig]
