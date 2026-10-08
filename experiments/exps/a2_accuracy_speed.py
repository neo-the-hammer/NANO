"""A2 -- accuracy vs speed of every backend / self-energy combination on
one Hamiltonian-path device (H_charge_T, RGF Lake).

Four runs of the same device:
    CPU + eigen        reference (the stock code path)
    CPU + decimation
    GPU + eigen        self-energy per energy on the CPU, RGF on the GPU
    GPU + decimation   everything batched on the GPU (the GPU default)
Each is scored against the reference (T, charge, current) and timed
(steady-state: best call after the first, so GPU start-up is excluded).
The plot is the trade-off: error vs time.

Pass: every combination within 1e-5 of the reference in charge and T.
"""

from common import device_step, rel_to_peak, steady, fmt_e, fmt_t, positive, CPU_C, GPU_C, MUTED

ID = "A2"
TITLE = "Accuracy vs speed: backend x self-energy method"
KIND = "accuracy"
TOL = 1e-5

COMBOS = [("CPU + eigen", False, "eig"), ("CPU + decimation", False, "dec"),
          ("GPU + eigen", True, "eig"), ("GPU + decimation", True, "dec")]


def spec(quick):
    if quick:
        return {"kind": "hamiltonian", "size": [6, 16], "grid": [-1.99, 1.99, 0.02]}
    return {"kind": "hamiltonian", "size": [8, 32], "grid": [-1.995, 1.995, 0.01]}


def steps(ctx):
    s = spec(ctx["quick"])
    return [device_step(lab, s, gpu, sh, 2, {"combo": lab}) for lab, gpu, sh in COMBOS]


def analyze(recs, ctx):
    by = {r["combo"]: r for r in recs if "error" not in r}
    ref = by.get("CPU + eigen")
    rows, worst, gpu = [], 0.0, False
    for lab, g, _ in COMBOS:
        r = by.get(lab)
        if r is None or ref is None:
            rows.append([lab, "-", "-", "-", "-", "ERROR"])
            continue
        if g and r["gpu_used"]:
            gpu = True
        eT = rel_to_peak(ref["T"], r["T"])
        eq = rel_to_peak(ref["charge"], r["charge"])
        eI = abs(ref["current"] - r["current"]) / max(abs(ref["current"]), 1e-300)
        r["_err"] = max(eT, eq)
        if r is not ref:
            worst = max(worst, eT, eq)
        sp = steady(ref["times"]) / steady(r["times"])
        rows.append([lab + ("" if (not g or r["gpu_used"]) else " (fell back to CPU)"),
                     fmt_e(eT), fmt_e(eq), fmt_e(eI), fmt_t(steady(r["times"])).strip(),
                     "%.1fx" % sp])
    n_err = sum(1 for r in recs if "error" in r)
    if n_err or ref is None:
        verdict, head = "FAIL", "%d run(s) errored" % n_err
    else:
        verdict = "PASS" if worst < TOL else "WARN"
        if not gpu:
            verdict = "NO GPU"
        best = min((r for r in by.values()), key=lambda r: steady(r["times"]))
        head = ("worst deviation from CPU+eigen %s; fastest: %s (%.1fx)"
                % (fmt_e(worst), best["combo"], steady(ref["times"]) / steady(best["times"])))
    return {"verdict": verdict, "headline": head,
            "table": {"cols": ["combination", "T err", "charge err", "current err",
                               "time", "speed vs ref"], "rows": rows}}


def plot(recs, summ, ctx, plt):
    by = {r["combo"]: r for r in recs if "error" not in r}
    ref = by.get("CPU + eigen")
    if ref is None:
        return []
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("A2  Accuracy vs speed on a Hamiltonian-path device (%d x %d, %d energies)"
                 % (ref["n"], ref["Nc"], ref["NE"]), x=0.01, ha="left")
    a, b, c = ax
    cols = {"CPU + eigen": CPU_C, "CPU + decimation": "#7fb0e8",
            "GPU + eigen": "#f3a77c", "GPU + decimation": GPU_C}
    for lab, r in by.items():
        err = max(rel_to_peak(ref["T"], r["T"]), rel_to_peak(ref["charge"], r["charge"]))
        a.loglog([steady(r["times"])], positive([err if r is not ref else 1e-16]), "o", ms=9,
                 color=cols[lab], label=lab + (" (reference)" if r is ref else ""))
    a.set_xlabel("time per NEGF solve (s)")
    a.set_ylabel("error vs CPU + eigen")
    a.set_title("(a) trade-off: lower-left is better")
    a.axhline(TOL, color="#d03b3b", ls="--", lw=1)
    a.legend(fontsize=7)
    labs = [l for l, _, _ in COMBOS if l in by]
    se, so, ot = [], [], []
    for l in labs:
        p = by[l]["profile"]
        if p:
            tot, s, v = p[-1]
            se.append(s); so.append(v); ot.append(max(tot - s - v, 0))
        else:
            se.append(0); so.append(steady(by[l]["times"])); ot.append(0)
    x = range(len(labs))
    b.bar(x, se, color="#c9a227", label="contact self-energy")
    b.bar(x, so, bottom=se, color=CPU_C, label="NEGF solve (RGF)")
    b.bar(x, ot, bottom=[p + q for p, q in zip(se, so)], color=MUTED, label="other")
    b.set_xticks(list(x))
    b.set_xticklabels(labs, rotation=20, ha="right", fontsize=7)
    b.set_ylabel("seconds (last call)")
    b.set_title("(b) where the time goes")
    b.legend(fontsize=7)
    for l in labs:
        r = by[l]
        if r is ref:
            continue
        c.semilogy(r["E"], positive([abs(p - q) for p, q in zip(ref["T"], r["T"])]),
                   color=cols[l], lw=1, label=l)
    c.set_xlabel("E (eV)")
    c.set_ylabel("|T - T_ref|")
    c.set_title("(c) transmission error per energy")
    c.legend(fontsize=7)
    fig.tight_layout()
    return [fig]
