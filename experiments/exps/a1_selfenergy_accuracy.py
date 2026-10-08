"""A1 -- accuracy of the contact self-energy methods vs broadening eta.

The Hamiltonian path has two self-energy methods: the original eigen
(mode-matching) method selfH_new, CPU only, and decimation selfH_dec,
which the GPU batches.  Neither is "the truth", so each result is scored
by its residual in the lead's own fixed-point equation
    Σ = β [ (E + iη - H_cell - Σ)^-1 ]_00 β†,
which any exact self-energy satisfies.  Swept over eta for a graphene
ribbon lead and a random four-slice lead; the GPU's decimation is scored
the same way on the same energies.

Expected: eigen ~1e-12 everywhere; decimation degrades as eta -> 0 (its
convergence slows); GPU decimation tracks CPU decimation.  Pass: GPU
decimation residual <= 10x CPU decimation residual (+1e-9) at every eta.
"""

from common import u_decim, u_selfh, Step, run_native, fmt_e, positive, CPU_C, GPU_C, ALT_C, MUTED

ID = "A1"
TITLE = "Self-energy accuracy vs eta: eigen vs decimation"
KIND = "accuracy"


def etas(quick):
    return [1e-3, 1e-5, 1e-7] if quick else [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]


def steps(ctx):
    n, NE = (4, 16) if ctx["quick"] else (6, 40)
    out = []
    for eta in etas(ctx["quick"]):
        for kind in ("gnr", "random"):
            a = ["selfh", "src", n, NE, eta, kind]
            out.append(Step("eig vs dec (CPU) %s eta=%g" % (kind, eta), "selfh", u_selfh(n, NE),
                            (lambda a=a: run_native(ctx["exe"], a))))
        # same random lead, same energies, through the batched GPU decimation
        a = ["decim", "hsrc", n, NE, eta, 1]
        out.append(Step("dec CPU vs GPU random eta=%g" % eta, "decim",
                        u_decim(n, NE) * (1 + (2 * 0.5 if ctx["gpu"] else 0)),
                        (lambda a=a: run_native(ctx["exe"], a)), ctx["gpu"]))
    return out


def series(recs):
    """eta -> {method: max residual}"""
    out = {}
    for r in recs:
        if "error" in r:
            continue
        e = out.setdefault(r["eta"], {})
        if r["cmd"] == "selfh":
            e["eig " + r["kind"]] = r["residual_eig"]
            e["dec CPU " + r["kind"]] = r["residual_dec"]
            e["t_eig"] = r["t_eig"] / r["NE"]
            e["t_dec"] = r["t_dec"] / r["NE"]
        else:
            e["dec CPU (batched) random"] = r["residual_cpu"]
            if r.get("gpu"):
                e["dec GPU random"] = r["residual_gpu"]
                e["t_dec_gpu"] = r["t_gpu"] / r["NB"]
    return out


def analyze(recs, ctx):
    S = series(recs)
    rows, bad, gpu = [], [], False
    for eta in sorted(S, reverse=True):
        e = S[eta]
        g = e.get("dec GPU random")
        c = e.get("dec CPU (batched) random")
        if g is not None:
            gpu = True
            if not g <= 10 * c + 1e-9:
                bad.append(eta)
        rows.append(["%g" % eta, fmt_e(e.get("eig gnr")), fmt_e(e.get("dec CPU gnr")),
                     fmt_e(e.get("eig random")), fmt_e(e.get("dec CPU random")), fmt_e(g)])
    n_err = sum(1 for r in recs if "error" in r)
    if n_err:
        verdict, head = "FAIL", "%d run(s) errored" % n_err
    else:
        verdict = ("FAIL" if bad else "PASS") if gpu else "NO GPU"
        e5 = S.get(1e-5, S[sorted(S)[len(S) // 2]])
        head = ("at eta=1e-5 residual: eigen %s, decimation %s (CPU) / %s (GPU)"
                % (fmt_e(e5.get("eig random")), fmt_e(e5.get("dec CPU random")),
                   fmt_e(e5.get("dec GPU random"))))
    return {"verdict": verdict, "headline": head,
            "table": {"cols": ["eta", "eig (GNR)", "dec (GNR)", "eig (random)",
                               "dec CPU (random)", "dec GPU (random)"], "rows": rows}}


def plot(recs, summ, ctx, plt):
    S = series(recs)
    ets = sorted(S)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("A1  Contact self-energy accuracy (residual of the lead equation)",
                 x=0.01, ha="left")
    a, b, c = ax
    styles = [("eig gnr", CPU_C, "-", "o"), ("dec CPU gnr", CPU_C, "--", "s"),
              ("eig random", "#3a9d5d", "-", "o"), ("dec CPU random", "#3a9d5d", "--", "s"),
              ("dec GPU random", GPU_C, ":", "^")]
    for key, col, ls, mk in styles:
        xs = [e for e in ets if S[e].get(key) is not None]
        if xs:
            a.loglog(xs, positive([S[e][key] for e in xs]), ls=ls, marker=mk, color=col, ms=5, label=key)
    a.set_xlabel("eta (eV)")
    a.set_ylabel("max residual over energies")
    a.set_title("(a) accuracy vs broadening")
    a.legend(fontsize=7)
    mid = 1e-5 if 1e-5 in S else ets[len(ets) // 2]
    for r in recs:
        if "error" in r or r["eta"] != mid:
            continue
        if r["cmd"] == "selfh":
            col = CPU_C if r["kind"] == "gnr" else "#3a9d5d"
            b.semilogy(r["E"], positive(r["res_eig"]), color=col, lw=1, label="eig " + r["kind"])
            b.semilogy(r["E"], positive(r["res_dec"]), color=col, lw=1, ls="--", label="dec " + r["kind"])
        elif r.get("gpu"):
            b.semilogy(r["E"], positive(r["res_gpu"]), "^", ms=3, color=GPU_C, label="dec GPU random")
    b.set_xlabel("E (eV)")
    b.set_ylabel("residual")
    b.set_title("(b) per energy at eta = %g" % mid)
    b.legend(fontsize=7)
    keys = [("t_eig", "eigen, CPU", CPU_C), ("t_dec", "decimation, CPU", "#3a9d5d"),
            ("t_dec_gpu", "decimation, GPU (batched)", GPU_C)]
    for k, (key, lab, col) in enumerate(keys):
        xs = [e for e in ets if S[e].get(key) is not None]
        c.bar([ets.index(e) + (k - 1) * 0.27 for e in xs], [1e3 * S[e][key] for e in xs], 0.27,
              color=col, label=lab)
    c.set_xticks(range(len(ets)))
    c.set_xticklabels(["%g" % e for e in ets])
    c.set_yscale("log")
    c.set_xlabel("eta (eV)")
    c.set_ylabel("ms per energy")
    c.set_title("(c) cost per energy")
    c.legend(fontsize=7)
    fig.tight_layout()
    return [fig]
