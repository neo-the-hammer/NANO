"""C2 -- contact self-energy (Sancho-Rubio decimation), CPU vs GPU.

vides_decimation_batch_cpu (the stock Gzerozero per energy) against the
batched CUDA decimation, on the three lead cells the code builds: the GNR
lead (selfGNR_cell) and the source and drain leads of the Hamiltonian
path (selfH_dec_cell, random four-slice periodic chain).

Decimation is iterative, and the GPU iterates until every energy of the
batch has converged, so an energy can see a few more iterations there;
near band edges the result is only as accurate as the conditioning
allows, so the two need not agree to round-off.  What matters is that both solve the
lead's own fixed-point equation equally well, so each result's residual
in that equation is computed too (a method-independent accuracy measure).

Pass: diff < 1e-8, or the GPU residual is no worse than 10x the CPU's and
the diff is within 10x the residual (both are equally good solutions).
"""

from common import with_gpu, u_decim, Step, run_native, fmt_e, positive, CPU_C, GPU_C, ALT_C, MUTED

ID = "C2"
TITLE = "Contact self-energy (decimation): CPU vs GPU"
KIND = "correctness"


def cases(quick):
    if quick:
        return [("gnr", 6, 32), ("gnr", 12, 32), ("hsrc", 6, 32), ("hdrn", 6, 32)]
    return [("gnr", 6, 64), ("gnr", 12, 64), ("gnr", 24, 64),
            ("hsrc", 6, 64), ("hsrc", 12, 64), ("hdrn", 6, 64), ("hdrn", 12, 64)]


def steps(ctx):
    out = []
    for cell, n, NB in cases(ctx["quick"]):
        args = ["decim", cell, n, NB, 1e-5, 1]
        units = with_gpu(u_decim(n, NB), 2, ctx["gpu"])
        out.append(Step("%s n=%d NB=%d" % (cell, n, NB), "decim", units,
                        (lambda a=args: run_native(ctx["exe"], a)), ctx["gpu"]))
    return out


def label(r):
    return "%s n=%d" % (r["cell"], r["n"])


def case_ok(r):
    d, rc, rg = r["worst_rel_diff"], r["residual_cpu"], r["residual_gpu"]
    if d is None or rg is None:
        return False
    return d < 1e-8 or (rg <= 10 * rc + 1e-9 and d <= 10 * max(rc, rg) + 1e-9)


def analyze(recs, ctx):
    ok = [r for r in recs if "error" not in r]
    gpu = any(r.get("gpu") for r in ok)
    rows = []
    for r in ok:
        g = r.get("gpu")
        rows.append([label(r), fmt_e(r["worst_rel_diff"]) if g else "-",
                     fmt_e(r["residual_cpu"]), fmt_e(r["residual_gpu"]) if g else "-",
                     "%.3f s" % r["t_cpu"], "%.3f s" % r["t_gpu"] if g else "-",
                     ("PASS" if case_ok(r) else "FAIL") if g else "NO GPU"])
    if not gpu:
        verdict, head = "NO GPU", "no CUDA device: only the CPU reference ran"
    elif len(ok) < len(recs):
        verdict, head = "FAIL", "%d case(s) errored" % (len(recs) - len(ok))
    else:
        bad = [r for r in ok if not case_ok(r)]
        verdict = "FAIL" if bad else "PASS"
        wd = max(r["worst_rel_diff"] for r in ok)
        wr = max(max(r["residual_cpu"], r["residual_gpu"]) for r in ok)
        head = ("worst CPU/GPU diff %s; worst lead residual %s (CPU and GPU alike)"
                % (fmt_e(wd), fmt_e(wr)))
    return {"verdict": verdict, "headline": head,
            "table": {"cols": ["lead cell", "rel. diff", "residual CPU", "residual GPU",
                               "CPU", "GPU", "verdict"], "rows": rows}}


def plot(recs, summ, ctx, plt):
    ok = [r for r in recs if "error" not in r and r.get("gpu")]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("C2  Contact self-energy by decimation: CPU vs GPU", x=0.01, ha="left")
    a, b, c = ax
    for i, r in enumerate(ok):
        a.semilogy(r["E"], positive(r["rel_diff"]), lw=1, color=ALT_C[i % len(ALT_C)], label=label(r))
    a.set_xlabel("E (eV)")
    a.set_ylabel("|Σ_cpu - Σ_gpu| / |Σ|")
    a.set_title("(a) difference per energy")
    a.legend(fontsize=7)
    lo, hi = 1e-16, 1e-16
    for i, r in enumerate(ok):
        xs, ys = positive(r["res_cpu"]), positive(r["res_gpu"])
        b.loglog(xs, ys, "o", ms=3, color=ALT_C[i % len(ALT_C)], label=label(r))
        hi = max([hi] + [v for v in xs + ys if v == v])
    b.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", label="equal accuracy")
    b.set_xlabel("lead-equation residual, CPU")
    b.set_ylabel("lead-equation residual, GPU")
    b.set_title("(b) accuracy: each point one energy")
    b.legend(fontsize=7)
    x = range(len(ok))
    c.bar([i - 0.2 for i in x], positive([r["worst_rel_diff"] for r in ok]), 0.4,
          color=GPU_C, label="worst CPU/GPU diff")
    c.bar([i + 0.2 for i in x], positive([max(r["residual_cpu"], r["residual_gpu"]) for r in ok]),
          0.4, color=CPU_C, label="worst residual")
    c.set_yscale("log")
    c.set_xticks(list(x))
    c.set_xticklabels([label(r) for r in ok], rotation=30, ha="right", fontsize=7)
    c.set_title("(c) diff is at the method's own accuracy")
    c.legend(fontsize=7)
    fig.tight_layout()
    return [fig]
