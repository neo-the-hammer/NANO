"""S3 -- whole-device speed, CPU vs GPU, with a time breakdown.

Every device path at two sizes, through the Python API, each backend in
its own process.  The GPU process solves twice and the second call is
reported (steady state: what each further step of a self-consistent loop
costs); the CPU solves once.  VIDES_PROFILE splits each solve into
contact self-energy, batched NEGF solve, and the rest (charge assembly,
Python and data movement).  Each backend uses its own defaults, as a user
would run it.
"""

from common import device_step, rel_to_peak, steady, fmt_e, fmt_t, CPU_C, GPU_C, MUTED

ID = "S3"
TITLE = "Whole-device speed with time breakdown"
KIND = "speed"


def devices(quick):
    if quick:
        g = [-1.505, 1.505, 0.01]
        return [("cnt", {"kind": "cnt", "size": [13, 5.0], "grid": g}),
                ("gnr", {"kind": "gnr", "size": [8, 5.0], "grid": g}),
                ("hamiltonian", {"kind": "hamiltonian", "size": [8, 24], "grid": g})]
    g = [-1.505, 1.505, 0.005]
    return [("cnt", {"kind": "cnt", "size": [10, 5.0], "grid": g}),
            ("cnt", {"kind": "cnt", "size": [19, 10.0], "grid": g}),
            ("cntmode", {"kind": "cntmode", "size": [19, 10.0], "grid": g, "nmodes": 6}),
            ("gnr", {"kind": "gnr", "size": [6, 5.0], "grid": g}),
            ("gnr", {"kind": "gnr", "size": [12, 10.0], "grid": g}),
            ("hamiltonian", {"kind": "hamiltonian", "size": [8, 40], "grid": g}),
            ("hamiltonian", {"kind": "hamiltonian", "size": [16, 40], "grid": g})]


def tag(name, spec):
    return "%s %s" % (name, "x".join(str(x) for x in spec["size"]))


def steps(ctx):
    out = []
    for name, spec in devices(ctx["quick"]):
        t = tag(name, spec)
        out.append(device_step(t + " CPU", spec, False, None, 1, {"device": t}))
        out.append(device_step(t + " GPU", spec, True, None, 2, {"device": t}))
    return out


def pairs(recs):
    by, order = {}, []
    for r in recs:
        if "error" in r:
            continue
        if r["device"] not in by:
            order.append(r["device"])
        by.setdefault(r["device"], {})["gpu" if r["gpu_requested"] else "cpu"] = r
    return [(k, by[k]["cpu"], by[k]["gpu"]) for k in order if len(by[k]) == 2]


def split(r):
    t = steady(r["times"])
    if r["profile"]:
        tot, s, v = r["profile"][-1]
        return t, s, v, max(t - s - v, 0)
    return t, 0.0, t, 0.0


def analyze(recs, ctx):
    rows, best, gpu = [], None, False
    for name, c, g in pairs(recs):
        tc, sc, vc, oc = split(c)
        tg, sg, vg, og = split(g)
        sp = tc / tg
        gpu = gpu or g["gpu_used"]
        if g["gpu_used"] and (best is None or sp > best[0]):
            best = (sp, name)
        rows.append([name, "%d x %d" % (c["n"], c["Nc"]), c["NE"], fmt_t(tc).strip(),
                     fmt_t(g["times"][0]).strip(), fmt_t(tg).strip(), "%.1fx" % sp,
                     "%.0f%%" % (100 * sc / tc if tc else 0), "%.0f%%" % (100 * sg / tg if tg else 0),
                     fmt_e(rel_to_peak(c["charge"], g["charge"]))])
    n_err = sum(1 for r in recs if "error" in r)
    if n_err:
        verdict, head = "FAIL", "%d run(s) errored" % n_err
    elif not gpu:
        verdict, head = "NO GPU", "GPU runs fell back to the CPU"
    else:
        verdict, head = "INFO", "best whole-device speedup %.1fx (%s)" % best
    return {"verdict": verdict, "headline": head,
            "table": {"cols": ["device", "n x Nc", "NE", "CPU", "GPU first", "GPU steady",
                               "speedup", "CPU Σ%", "GPU Σ%", "charge diff"], "rows": rows}}


def plot(recs, summ, ctx, plt):
    P = pairs(recs)
    if not P:
        return []
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [3, 1.3]})
    fig.suptitle("S3  Whole devices: time per NEGF solve, CPU vs GPU", x=0.01, ha="left")
    a, b = ax
    w = 0.38
    for i, (name, c, g) in enumerate(P):
        for k, (r, col) in enumerate([(c, CPU_C), (g, GPU_C)]):
            t, s, v, o = split(r)
            x = i + (k - 0.5) * w
            a.bar(x, s, w, color="#c9a227", edgecolor=col, linewidth=1.5,
                  label="contact self-energy" if i == 0 and k == 0 else None)
            a.bar(x, v, w, bottom=s, color=col, label=("NEGF solve, CPU", "NEGF solve, GPU")[k]
                  if i == 0 else None)
            a.bar(x, o, w, bottom=s + v, color=MUTED, edgecolor=col, linewidth=1.5,
                  label="other" if i == 0 and k == 0 else None)
        tc, tg = split(c)[0], split(g)[0]
        a.text(i, max(tc, tg) * 1.02, "%.1fx" % (tc / tg), ha="center", fontsize=8,
               fontweight="bold")
    a.set_xticks(range(len(P)))
    a.set_xticklabels([p[0] for p in P], rotation=20, ha="right", fontsize=8)
    a.set_ylabel("seconds per solve")
    a.set_title("(a) breakdown (left bar CPU, right bar GPU; label = speedup)")
    a.legend(fontsize=7)
    sps = [split(c)[0] / split(g)[0] for _, c, g in P]
    b.barh(range(len(P)), sps, color=[GPU_C if s >= 1 else MUTED for s in sps])
    b.axvline(1, color="#d03b3b", lw=1, ls="--")
    b.set_yticks(range(len(P)))
    b.set_yticklabels([p[0] for p in P], fontsize=8)
    b.invert_yaxis()
    b.set_xlabel("speedup CPU / GPU (steady)")
    b.set_title("(b) speedup")
    fig.tight_layout()
    return [fig]
