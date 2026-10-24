"""S4 -- CPU thread scaling of the three CPU kernels.

The CPU paths now spread energies over threads (VIDES_THREADS, default
all cores).  Swept over the thread count 1, 2, 4, ... up to the cores
online, at two problem sizes each:

  rgf     the NEGF solver (LDOS per energy, as the stock code)
  decim   the decimation self-energy (Gzerozero per energy)
  selfh   the eigen self-energy (selfH_new per energy): the CPU-only part
          of the Hamiltonian path that dominated its GPU runs (S3: 81-93%)

Speedup = time with 1 thread / time with T threads; the dashed line is
ideal scaling.  Every thread count must reproduce the 1-thread result
exactly (the "max diff" column; anything but 0 is a FAIL), so threading
is a pure speedup with no change in the numbers.
"""

from common import Step, run_native, u_rgf, u_decim, u_selfh, cpu_count, fmt_t, CPU_C, GPU_C, ALT_C

ID = "S4"
TITLE = "CPU thread scaling (solver, decimation, eigen self-energy)"
KIND = "speed"


def tlist():
    n = cpu_count()
    t, v = [], 1
    while v < n:
        t.append(v)
        v *= 2
    t.append(n)
    return t


def cases(quick):
    if quick:
        return [("rgf", 12, 30, 64), ("rgf", 24, 30, 64), ("decim", 6, 0, 32), ("decim", 12, 0, 16),
                ("selfh", 6, 0, 32), ("selfh", 12, 0, 32)]
    return [("rgf", 12, 40, 128), ("rgf", 32, 40, 128), ("decim", 8, 0, 64), ("decim", 16, 0, 32),
            ("selfh", 8, 0, 64), ("selfh", 16, 0, 64)]


def steps(ctx):
    out, T = [], tlist()
    tl = ",".join(str(t) for t in T)
    for kind, n, Nc, NB in cases(ctx["quick"]):
        u = {"rgf": lambda: u_rgf(n, Nc, NB), "decim": lambda: u_decim(n, NB),
             "selfh": lambda: u_selfh(n, NB) / 4}[kind]() * sum(1.0 / t for t in T)
        a = ["threads", kind, n, Nc, NB, 1e-5, tl]
        cls = {"rgf": "rgf", "decim": "decim", "selfh": "selfh"}[kind]
        out.append(Step("%s n=%d NB=%d, threads %s" % (kind, n, NB, tl), cls, u,
                        (lambda a=a: run_native(ctx["exe"], a))))
    return out


def analyze(recs, ctx):
    rows, bad, best = [], 0, None
    for r in recs:
        if "error" in r:
            continue
        t = r["t"]
        sp = [t[0] / x if x > 0 else float("nan") for x in t]
        if r["maxdiff_vs_1"] != 0:
            bad += 1
        k = len(t) - 1
        eff = sp[k] / r["thread_counts"][k]
        if best is None or sp[k] > best[0]:
            best = (sp[k], r)
        rows.append([r["kind"], r["n"], r["NB"], " / ".join(fmt_t(x).strip() for x in t),
                     " / ".join("%.1fx" % x for x in sp), "%.0f%%" % (100 * eff),
                     "0 (exact)" if r["maxdiff_vs_1"] == 0 else "%.1e" % r["maxdiff_vs_1"]])
    n_err = sum(1 for r in recs if "error" in r)
    T = recs[0]["thread_counts"] if recs and "error" not in recs[0] else tlist()
    if n_err:
        verdict, head = "FAIL", "%d run(s) errored" % n_err
    elif bad:
        verdict, head = "FAIL", "%d case(s) changed the result with threads" % bad
    else:
        verdict = "INFO"
        head = "best %.1fx on %d threads (%s n=%d); results identical to 1 thread" % (
            best[0], T[-1], best[1]["kind"], best[1]["n"])
    return {"verdict": verdict, "headline": head,
            "table": {"cols": ["kernel", "n", "NB", "time at threads %s" % "/".join(map(str, T)),
                               "speedup", "efficiency (max threads)", "max diff vs 1 thread"],
                      "rows": rows}}


def plot(recs, summ, ctx, plt):
    ok = [r for r in recs if "error" not in r]
    if not ok:
        return []
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    fig.suptitle("S4  CPU thread scaling: speedup over 1 thread (dashed: ideal)", x=0.01, ha="left")
    for a, kind, title in zip(ax, ("rgf", "decim", "selfh"),
                              ("NEGF solver", "decimation self-energy", "eigen self-energy")):
        rs = [r for r in ok if r["kind"] == kind]
        for i, r in enumerate(rs):
            T = r["thread_counts"]
            a.plot(T, [r["t"][0] / x for x in r["t"]], "o-", color=(CPU_C, GPU_C, ALT_C[2])[i % 3],
                   label="n=%d" % r["n"])
        if rs:
            T = rs[0]["thread_counts"]
            a.plot(T, T, "--", color="#898781", lw=1)
        a.set_title(title)
        a.set_xlabel("CPU threads")
        a.set_ylabel("speedup")
        a.legend(fontsize=7)
    fig.tight_layout()
    return [fig]
