"""S2 -- speed of the contact self-energy, every method and backend.

  decimation, GPU   batched over energies (vides_decimation_batch_gpu)
  decimation, CPU   Gzerozero per energy (stock selfGNR / selfH_dec)
  eigen, CPU        selfH_new per energy (the Hamiltonian path's default
                    on a CPU)
Swept over lead width n at fixed batch, and over batch size NB at fixed n.
Times are per energy so the methods can be read off one axis.
"""

from common import with_gpu, u_decim, u_selfh, Step, run_native, fmt_t, CPU_C, GPU_C, MUTED

ID = "S2"
TITLE = "Self-energy speed: eigen vs decimation, CPU vs GPU"
KIND = "speed"


def plan(quick):
    if quick:
        return [4, 8, 12], 32, 8, [8, 32, 128], 12
    return [4, 8, 12, 16], 64, 8, [16, 64, 256], 16


def steps(ctx):
    ns, NB, nfix, NBs, NE = plan(ctx["quick"])
    out = []
    g = ctx["gpu"]
    for n in ns:
        a = ["decim", "hsrc", n, NB, 1e-5, 2]
        out.append(Step("decimation CPU+GPU n=%d NB=%d" % (n, NB), "decim",
                        with_gpu(u_decim(n, NB), 3, g),
                        (lambda a=a: dict(run_native(ctx["exe"], a), axis="n")), g))
        a = ["selfh", "src", n, NE, 1e-5, "random"]
        out.append(Step("eigen vs decimation CPU n=%d" % n, "selfh", u_selfh(n, NE),
                        (lambda a=a: dict(run_native(ctx["exe"], a), axis="n"))))
    for NBv in NBs:
        a = ["decim", "hsrc", nfix, NBv, 1e-5, 2]
        out.append(Step("decimation CPU+GPU n=%d NB=%d" % (nfix, NBv), "decim",
                        with_gpu(u_decim(nfix, NBv), 3, g),
                        (lambda a=a: dict(run_native(ctx["exe"], a), axis="NB")), g))
    return out


def per_energy(recs):
    """n -> {method: seconds per energy} for the n sweep"""
    S = {}
    for r in recs:
        if "error" in r or r["axis"] != "n":
            continue
        e = S.setdefault(r["n"], {})
        if r["cmd"] == "decim":
            e["dec CPU"] = r["t_cpu"] / r["NB"]
            if r.get("gpu"):
                e["dec GPU"] = r["t_gpu"] / r["NB"]
        else:
            e["eig CPU"] = r["t_eig"] / r["NE"]
            e.setdefault("dec CPU (per call)", r["t_dec"] / r["NE"])
    return S


def analyze(recs, ctx):
    S = per_energy(recs)
    rows = []
    for n in sorted(S):
        e = S[n]
        g = e.get("dec GPU")
        rows.append([n, 4 * n, fmt_t(e.get("eig CPU")).strip(), fmt_t(e.get("dec CPU")).strip(),
                     fmt_t(g).strip() if g else "-",
                     "%.1fx" % (e["dec CPU"] / g) if g else "-",
                     "%.1fx" % (e["eig CPU"] / g) if g and "eig CPU" in e else "-"])
    nb = [r for r in recs if "error" not in r and r["axis"] == "NB"]
    for r in sorted(nb, key=lambda r: r["NB"]):
        if r.get("gpu"):
            rows.append(["n=%d" % r["n"], "NB=%d" % r["NB"], "-",
                         fmt_t(r["t_cpu"] / r["NB"]).strip(), fmt_t(r["t_gpu"] / r["NB"]).strip(),
                         "%.1fx" % (r["t_cpu"] / r["t_gpu"]), "-"])
    n_err = sum(1 for r in recs if "error" in r)
    gpu = any(r.get("gpu") for r in recs if "error" not in r)
    if n_err:
        verdict, head = "FAIL", "%d run(s) errored" % n_err
    elif not gpu:
        verdict, head = "NO GPU", "CPU timings only"
    else:
        nmax = max(S)
        e = S[nmax]
        verdict = "INFO"
        head = ("n=%d: GPU decimation %.1fx faster than CPU decimation, %.1fx vs CPU eigen"
                % (nmax, e["dec CPU"] / e["dec GPU"], e.get("eig CPU", float("nan")) / e["dec GPU"]))
    return {"verdict": verdict, "headline": head,
            "table": {"cols": ["n", "cell (4n)", "eigen CPU / E", "dec CPU / E",
                               "dec GPU / E", "GPU vs dec CPU", "GPU vs eigen CPU"],
                      "rows": rows}}


def plot(recs, summ, ctx, plt):
    S = per_energy(recs)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("S2  Contact self-energy: time per energy", x=0.01, ha="left")
    a, b = ax
    ns = sorted(S)
    for key, col, mk, lab in [("eig CPU", CPU_C, "o", "eigen, CPU"),
                              ("dec CPU", "#3a9d5d", "s", "decimation, CPU"),
                              ("dec GPU", GPU_C, "^", "decimation, GPU (batched)")]:
        xs = [n for n in ns if key in S[n]]
        if xs:
            a.loglog(xs, [1e3 * S[n][key] for n in xs], marker=mk, color=col, label=lab)
    a.set_xlabel("lead slice width n (cell is 4n)")
    a.set_ylabel("ms per energy")
    a.set_title("(a) vs lead size")
    a.legend(fontsize=7)
    nb = sorted([r for r in recs if "error" not in r and r["axis"] == "NB"], key=lambda r: r["NB"])
    if nb:
        b.loglog([r["NB"] for r in nb], [1e3 * r["t_cpu"] / r["NB"] for r in nb], "s-",
                 color="#3a9d5d", label="decimation, CPU")
        g = [r for r in nb if r.get("gpu")]
        if g:
            b.loglog([r["NB"] for r in g], [1e3 * r["t_gpu"] / r["NB"] for r in g], "^-",
                     color=GPU_C, label="decimation, GPU")
        b.set_title("(b) vs energies per batch (n=%d)" % nb[0]["n"])
    b.set_xlabel("energies per batch NB")
    b.set_ylabel("ms per energy")
    b.legend(fontsize=7)
    fig.tight_layout()
    return [fig]
