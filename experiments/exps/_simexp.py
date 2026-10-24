"""Shared steps / analysis / plots for the experiments built on sims.py
(R1 tutorials, R2 regression tests, P1 device physics).

Every scenario runs twice, in fresh processes with VIDES_GPU=0 and
VIDES_GPU=1, each backend with its own defaults (as a user would run the
script).  A scenario is judged on three things:

  agreement   every series and scalar, CPU vs GPU: max|cpu - gpu| /
              max(|cpu|, |gpu|) against the scenario's tolerance; WARN up
              to 100x, FAIL beyond
  references  error against the stored result shipped in test/, on each
              backend, against the original test's own tolerance
  laws        physics checks (0 <= T <= channels, symmetries, monotonic
              I-V, ...), on each backend

A reference or a law that passes on the CPU and fails on the GPU is a
FAIL.  One that fails on both backends alike says nothing about the GPU
(the reference is out of date, or the expectation does not hold for that
device) and is reported as WARN.
"""

from common import Step, run_sim, rel_to_peak, fmt_e, fmt_t, positive, CPU_C, GPU_C, MUTED, finite

TINY = 1e-300


def make_steps(scen, ctx):
    """scen: list of (name, tol, (cpu_seconds_quick, cpu_seconds_full))."""
    out = []
    for name, tol, secs in scen:
        s = secs[0] if ctx["quick"] else secs[1]
        for gpu in (False, True):
            def fn(name=name, gpu=gpu):
                r = run_sim(name, gpu, ctx["quick"])
                r["scenario"] = name
                return r
            # units in nominal CPU-seconds x 1e9: scaled by the calibrated
            # speed of this machine, refined as scenarios finish
            out.append(Step("%s %s" % (name, "GPU" if gpu else "CPU"),
                            "sim_%s_%s" % (name.replace("_", ""), "gpu" if gpu else "cpu"),
                            s * 1e9, fn, gpu))
    return out


def pairs(recs, scen):
    by = {}
    for r in recs:
        if "scenario" in r:
            by.setdefault(r["scenario"], {})["gpu" if r["gpu_requested"] else "cpu"] = r
        elif "label" in r:             # a step that raised before returning
            name, _, be = r["label"].rpartition(" ")
            by.setdefault(name, {})[be.lower()] = r
    return [(n, t, by.get(n, {}).get("cpu"), by.get(n, {}).get("gpu")) for n, t, _ in scen
            if n in by]


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), TINY)


def diffs(c, g):
    """{quantity: rel. diff} over the series and scalars both runs hold."""
    d = {}
    for k in c.get("series", {}):
        if k in g.get("series", {}):
            d[k] = rel_to_peak(c["series"][k], g["series"][k])
    for k in c.get("scalars", {}):
        if k in g.get("scalars", {}):
            d[k] = rel(c["scalars"][k], g["scalars"][k])
    return d


def checks(c, g, key):
    """[(name, what, tol, err_cpu, err_gpu, status)] for refs or laws."""
    out = []
    for k, v in c.get(key, {}).items():
        gv = g.get(key, {}).get(k) if g else None
        ec, eg = v["err"], (gv["err"] if gv else None)
        okc = finite(ec) and ec <= v["tol"]
        okg = gv is not None and finite(eg) and eg <= v["tol"]
        if gv is None:
            st = "CPU only"
        elif okc and okg:
            st = "PASS"
        elif okc and not okg:
            st = "FAIL"            # the GPU broke what the CPU gets right
        elif okg and not okc:
            st = "WARN"            # GPU fine, CPU off: not a GPU problem
        else:
            st = "WARN"            # off on both alike
        out.append((k, v.get("what", ""), v["tol"], ec, eg, st))
    return out


def judge(name, tol, c, g):
    """(verdict, worst_quantity, worst_diff, ref_rows, law_rows, note)"""
    if c is None or g is None:
        return "FAIL", None, None, [], [], "a run is missing"
    if "skipped" in c or "skipped" in g:
        return "SKIP", None, None, [], [], (c.get("skipped") or g.get("skipped"))
    ec, eg = "error" in c, "error" in g
    if ec or eg:
        if ec and eg:
            return "ERROR", None, None, [], [], "fails on both backends: " + _last(c["error"])
        return "FAIL", None, None, [], [], "fails on the %s only: %s" % (
            "CPU" if ec else "GPU", _last((c if ec else g)["error"]))
    d = diffs(c, g)
    wq, wd = (max(d.items(), key=lambda kv: kv[1] if finite(kv[1]) else float("inf"))
              if d else (None, 0.0))
    refs, laws = checks(c, g, "refs"), checks(c, g, "laws")
    note = ""
    if not g.get("gpu_used"):
        return "NO GPU", wq, wd, refs, laws, "GPU run fell back to the CPU"
    v = "PASS" if finite(wd) and wd < tol else ("WARN" if finite(wd) and wd < 100 * tol else "FAIL")
    st = [r[-1] for r in refs + laws]
    if "FAIL" in st:
        v = "FAIL"
        note = "a check passes on the CPU and fails on the GPU"
    elif "WARN" in st and v == "PASS":
        v = "WARN"
        note = "a check is off on both backends alike (not the GPU's)"
    ic, ig = c.get("info", {}).get("SCF iterations"), g.get("info", {}).get("SCF iterations")
    if ic is not None and ic != ig:
        note = (note + "; " if note else "") + "SCF iterations differ: CPU %s, GPU %s" % (ic, ig)
    return v, wq, wd, refs, laws, note


def _last(tb):
    lines = [l for l in str(tb).strip().splitlines() if l.strip()]
    return lines[-1][:200] if lines else "?"


def _count(rows):
    if not rows:
        return "-"
    ok = sum(1 for r in rows if r[-1] == "PASS")
    return "%d/%d" % (ok, len(rows))


def analyze(recs, ctx, scen, label):
    rows, detail, verdicts, notes = [], [], [], []
    for name, tol, c, g in pairs(recs, scen):
        v, wq, wd, refs, laws, note = judge(name, tol, c, g)
        verdicts.append(v)
        if note:
            notes.append("%s: %s" % (name, note))
        tc = c.get("wall") if c and "wall" in c else None
        tg = g.get("wall") if g and "wall" in g else None
        rows.append([name, "%s/%s" % ((c or {}).get("solves", "-"), (g or {}).get("solves", "-")),
                     fmt_t(tc).strip(), fmt_t(tg).strip(),
                     "%.1fx" % (tc / tg) if tc and tg else "-",
                     fmt_e(wd) if wd is not None else "-", (wq or "-")[:28], "%g" % tol,
                     _count(refs), _count(laws), v])
        for kind, rr in (("ref", refs), ("law", laws)):
            for k, what, t, ec, eg, st in rr:
                detail.append([name, kind, k[:40], fmt_e(ec), fmt_e(eg) if eg is not None else "-",
                               "%g" % t, st, what[:60]])
        if c and g and "series" in c and "series" in g:
            for k, dv in sorted(diffs(c, g).items()):
                detail.append([name, "diff", k[:40], "", "", "%g" % tol,
                               "PASS" if finite(dv) and dv < tol else
                               ("WARN" if finite(dv) and dv < 100 * tol else "FAIL"),
                               "CPU vs GPU, rel. to peak: %s" % fmt_e(dv)])
    n = len(verdicts)
    if not n:
        verdict, head = "FAIL", "no scenario ran"
    elif all(v == "NO GPU" for v in verdicts if v not in ("SKIP",)) and "NO GPU" in verdicts:
        verdict, head = "NO GPU", "GPU runs fell back to the CPU (no CUDA device or CPU-only build)"
    else:
        if "FAIL" in verdicts:
            verdict = "FAIL"
        elif "WARN" in verdicts or "ERROR" in verdicts:
            verdict = "WARN"
        else:
            verdict = "PASS"
        cnt = {k: verdicts.count(k) for k in ("PASS", "WARN", "FAIL", "ERROR", "SKIP", "NO GPU")
               if verdicts.count(k)}
        head = "%s: %s" % (label, ", ".join("%d %s" % (v, k) for k, v in cnt.items()))
    out = {"verdict": verdict, "headline": head,
           "table": {"cols": ["scenario", "solves CPU/GPU", "CPU", "GPU", "speedup", "worst diff",
                              "in", "tol", "refs", "laws", "verdict"], "rows": rows},
           "tables": [{"title": "every check", "cols": ["scenario", "kind", "quantity", "CPU",
                                                       "GPU", "tol", "status", "about"],
                       "rows": detail}]}
    if notes:
        out["notes"] = notes
    return out


def plot(recs, summ, ctx, plt, scen, title, per_row=3):
    P = [(n, t, c, g) for n, t, c, g in pairs(recs, scen)
         if c and g and "series" in c and "series" in g]
    if not P:
        return []
    fig, ax = plt.subplots(len(P), per_row + 1, figsize=(4.0 * (per_row + 1), 2.7 * len(P)),
                           squeeze=False)
    fig.suptitle("%s: CPU (solid) vs GPU (dashed); right: CPU-GPU difference per quantity"
                 % title, x=0.01, ha="left")
    for i, (name, tol, c, g) in enumerate(P):
        keys = [k for k in c["series"] if k in g["series"]]
        # prefer the curves a reader recognises: T, I-V, then the rest
        keys.sort(key=lambda k: (0 if ("I(" in k or "Id(" in k) else 1 if "T" in k.split(":")[-1][:3]
                                 else 2, k))
        for j in range(per_row):
            a = ax[i][j]
            if j >= len(keys):
                a.axis("off")
                continue
            k = keys[j]
            yc, yg = c["series"][k], g["series"][k]
            xs = c.get("x", {}).get(k) or list(range(len(yc)))
            xg = g.get("x", {}).get(k) or list(range(len(yg)))
            mk = "o" if len(yc) < 30 else None
            a.plot(xs, yc, color=CPU_C, lw=1.6, marker=mk, ms=4, label="CPU")
            a.plot(xg, yg, color=GPU_C, lw=1.0, ls="--", marker="x" if mk else None, ms=5, label="GPU")
            a.set_title(k[:42], fontsize=8.5)
            if j == 0:
                a.set_ylabel(name, fontsize=8.5)
                if i == 0:
                    a.legend(fontsize=7)
        b = ax[i][per_row]
        d = diffs(c, g)
        ks = sorted(d)
        b.barh(range(len(ks)), positive([d[k] for k in ks]),
               color=[GPU_C if finite(d[k]) and d[k] < tol else "#d03b3b" for k in ks])
        b.set_xscale("log")
        b.axvline(tol, color="#d03b3b", lw=1, ls="--")
        b.set_yticks(range(len(ks)))
        b.set_yticklabels([k[:22] for k in ks], fontsize=6.5)
        b.invert_yaxis()
        b.set_title("rel. difference (dashed: tol %g)" % tol, fontsize=8.5)
    fig.tight_layout()
    figs = [fig]
    # second figure: every reference and law check, CPU vs GPU error / tol
    rows = []
    for name, tol, c, g in P:
        for kind in ("refs", "laws"):
            for k, what, t, ec, eg, st in checks(c, g, kind):
                rows.append(("%s | %s" % (name, k), ec / t if t else ec,
                             (eg / t if t else eg) if eg is not None else float("nan")))
    if rows:
        f2, a = plt.subplots(figsize=(10, max(2.5, 0.22 * len(rows) + 1)))
        f2.suptitle("%s: reference and physics checks (error / tolerance; < 1 passes)" % title,
                    x=0.01, ha="left")
        y = list(range(len(rows)))
        a.barh([v - 0.2 for v in y], positive([r[1] for r in rows], 1e-12), 0.4, color=CPU_C, label="CPU")
        a.barh([v + 0.2 for v in y], positive([r[2] for r in rows], 1e-12), 0.4, color=GPU_C, label="GPU")
        a.axvline(1, color="#d03b3b", lw=1, ls="--")
        a.set_xscale("log")
        a.set_yticks(y)
        a.set_yticklabels([r[0][:60] for r in rows], fontsize=6.5)
        a.invert_yaxis()
        a.legend(fontsize=7)
        f2.tight_layout()
        figs.append(f2)
    return figs
