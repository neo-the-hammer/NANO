#!/usr/bin/env python3
"""E5 -- a study the speedup makes practical: Anderson localization in
disordered graphene nanoribbons.

Statistics over many random devices is where a faster solver turns into
new results: every disorder configuration is a full NEGF calculation.
Armchair ribbons (n = 8, metallic) of several lengths L get on-site
Anderson disorder of strength W (each atom's energy uniform in
[-W/2, W/2]); for each (L, W) an ensemble of random ribbons is solved for
T(E) over 40 energies near the band centre (0.2 to 0.6 eV).

Physics checked:
  clean ribbon (W = 0): T is an integer (number of open channels)
  disorder: <ln T> falls linearly with L, <ln T> = -2 L / xi + c, the
    signature of localization; the localization length xi comes from the
    fit and must shrink as W grows
Numerics checked: the first samples are also solved on the CPU, with the
same random ribbons (seeded by L, W and the sample index), and must agree
with the GPU to 1e-7 relative to peak.  Speed: the GPU solves the whole
ensemble; the CPU time for it is projected from its per-device time.
"""

import math

import _runner  # noqa: F401
import _proc
from common import Step, u_device, fmt_t, fmt_e, CPU_C, GPU_C, ALT_C, MUTED

ID = "E5"
TITLE = "Science case: localization in disordered nanoribbons (device ensembles)"
KIND = "speed"
GRID = [0.2025, 0.5925, 0.01]
TOL = 1e-7


def plan(quick):
    if quick:
        return {"n": 8, "lengths": [5.0, 10.0], "W": [0.0, 1.0], "samples": 4, "cpu_samples": 2}
    return {"n": 8, "lengths": [5.0, 10.0, 15.0, 20.0, 30.0], "W": [0.0, 0.5, 1.0, 2.0],
            "samples": 24, "cpu_samples": 3}


def n_solves(p, samples):
    return sum(1 if W == 0 else samples for W in p["W"]) * len(p["lengths"])


def units(p, samples, gpu):
    per = sum(u_device({"kind": "gnr", "size": [p["n"], L], "grid": GRID}, gpu) for L in p["lengths"]) / len(p["lengths"])
    return per * n_solves(p, samples)


def steps(ctx):
    p = plan(ctx["quick"])
    pay = {"n": p["n"], "lengths": p["lengths"], "W": p["W"], "grid": GRID, "eta": 1e-6}
    out = []
    for gpu, samples in ((False, p["cpu_samples"]), (True, p["samples"])):
        pp = dict(pay, samples=samples)
        out.append(Step("ensemble %s (%d solves)" % ("GPU" if gpu else "CPU", n_solves(p, samples)),
                        "dev_gnr_%s" % ("gpu" if gpu else "cpu"), units(p, samples, gpu),
                        (lambda pp=pp, gpu=gpu: dict(_proc.run_worker("disorder_worker.py", pp, gpu),
                                                     gpu_req=gpu)), gpu))
    return out


def _median(v):
    v = sorted(v)
    return v[len(v) // 2] if v else float("nan")


def _fit(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if not sxx:
        return None, None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return b, my - b * mx


def ensemble(rec):
    """{W: {L: [ln T values over samples and energies]}} and clean-ribbon T."""
    by, clean = {}, []
    for s in rec["samples"]:
        if s["W"] == 0:
            clean.extend(s["T"])
            continue
        by.setdefault(s["W"], {}).setdefault(s["L"], []).extend(math.log(max(t, 1e-300)) for t in s["T"])
    return by, clean


def localization(by):
    """xi(W) from the slope of <ln T> against L."""
    out = {}
    for W, m in by.items():
        Ls = sorted(m)
        if len(Ls) < 2:
            continue
        mean = [sum(m[L]) / len(m[L]) for L in Ls]
        b, c = _fit(Ls, mean)
        out[W] = {"L": Ls, "mean": mean, "slope": b, "icpt": c, "xi": (-2.0 / b) if b and b < 0 else None}
    return out


def analyze(recs, ctx):
    c = next((r for r in recs if "error" not in r and not r.get("gpu_req")), None)
    g = next((r for r in recs if "error" not in r and r.get("gpu_req")), None)
    n_err = sum(1 for r in recs if "error" in r)
    if n_err or not c:
        return {"verdict": "FAIL", "headline": "%d run(s) errored" % n_err if n_err else "no CPU run",
                "table": None}
    main = g if g and g.get("gpu_used") else c
    # CPU vs GPU on the same random ribbons
    diff = None
    if g:
        gi = {(s["L"], s["W"], s["k"]): s["T"] for s in g["samples"]}
        diffs = []
        for s in c["samples"]:
            t2 = gi.get((s["L"], s["W"], s["k"]))
            if t2:
                pk = max(max(map(abs, s["T"])), 1e-12)
                diffs.append(max(abs(a - b) for a, b in zip(s["T"], t2)) / pk)
        diff = max(diffs) if diffs else None
    by, clean = ensemble(main)
    loc = localization(by)
    # laws
    laws = []
    nonint = max((abs(t - round(t)) for t in clean), default=0.0)
    laws.append(("clean ribbon: integer T", nonint, 1e-3))
    for W, f in sorted(loc.items()):
        drop = f["mean"][0] - f["mean"][-1]
        laws.append(("W=%g: <ln T> falls with L" % W, max(0.0, -drop), 1e-9))
    xis = [(W, f["xi"]) for W, f in sorted(loc.items()) if f["xi"]]
    mono = max((max(0.0, b[1] - a[1]) for a, b in zip(xis, xis[1:])), default=0.0)
    if len(xis) > 1:
        laws.append(("xi shrinks as W grows", mono, 1e-9))
    rows = []
    for W, f in sorted(loc.items()):
        for L, m in zip(f["L"], f["mean"]):
            rows.append(["%g" % W, "%g" % L, "%.3f" % m, "%.1f nm" % f["xi"] if f["xi"] else "-"])
    t_c = _median(c["times"])
    t_g = _median(g["times"]) if g else None
    p = plan(ctx["quick"])
    total = n_solves(p, p["samples"])
    proj_cpu = t_c * total
    t_ens = sum(g["times"]) if g else None
    bad = [l for l in laws if not (l[1] <= l[2])]
    gpu = bool(g and g.get("gpu_used"))
    if not gpu:
        verdict = "NO GPU" if not bad else "WARN"
    elif diff is not None and diff >= 100 * TOL:
        verdict = "FAIL"
    elif bad or (diff is not None and diff >= TOL):
        verdict = "WARN"
    else:
        verdict = "PASS"
    head = ("%d disordered ribbons: xi = %s; ensemble on the GPU %s vs %s projected on the CPU (%.1fx); "
            "CPU vs GPU on the same ribbons %s"
            % (total, ", ".join("%.0f nm (W=%g)" % (x, W) for W, x in xis) or "-",
               fmt_t(t_ens).strip() if t_ens else "-", fmt_t(proj_cpu).strip(),
               proj_cpu / t_ens if t_ens else float("nan"), fmt_e(diff)))
    return {"verdict": verdict, "headline": head,
            "notes": ["per device: CPU %s, GPU %s (median)" % (fmt_t(t_c).strip(), fmt_t(t_g).strip() if t_g else "-")],
            "table": {"cols": ["W (eV)", "L (nm)", "<ln T>", "localization length"], "rows": rows},
            "tables": [{"title": "checks", "cols": ["check", "value", "tol", "status"],
                        "rows": [[n, fmt_e(v), "%g" % t, "PASS" if v <= t else "WARN"] for n, v, t in laws] +
                        ([["CPU vs GPU, same ribbons", fmt_e(diff), "%g" % TOL,
                           "PASS" if diff < TOL else "FAIL"]] if diff is not None else [])}]}


def plot(recs, summ, ctx, plt):
    g = next((r for r in recs if "error" not in r and r.get("gpu_req") and r.get("gpu_used")), None) or \
        next((r for r in recs if "error" not in r), None)
    if not g or "samples" not in g:
        return []
    by, _ = ensemble(g)
    loc = localization(by)
    if not loc:
        return []
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    fig.suptitle("E5  Anderson localization in n=8 armchair ribbons (%d devices)" % len(g["samples"]),
                 x=0.01, ha="left")
    for k, (W, f) in enumerate(sorted(loc.items())):
        col = ALT_C[k % len(ALT_C)]
        ax[0].plot(f["L"], f["mean"], "o", color=col, label="W = %g eV" % W)
        if f["slope"] is not None:
            ax[0].plot(f["L"], [f["icpt"] + f["slope"] * L for L in f["L"]], "-", color=col, lw=1)
    ax[0].set_xlabel("ribbon length L (nm)")
    ax[0].set_ylabel("<ln T>")
    ax[0].set_title("(a) <ln T> falls linearly with L")
    ax[0].legend()
    xs = [(W, f["xi"]) for W, f in sorted(loc.items()) if f["xi"]]
    if xs:
        ax[1].loglog([x[0] for x in xs], [x[1] for x in xs], "o-", color=GPU_C)
    ax[1].set_xlabel("disorder W (eV)")
    ax[1].set_ylabel("localization length xi (nm)")
    ax[1].set_title("(b) localization length")
    Wmax = max(by)
    Lmax = max(by[Wmax])
    ax[2].hist(by[Wmax][Lmax], bins=30, color=MUTED)
    ax[2].set_xlabel("ln T")
    ax[2].set_ylabel("count")
    ax[2].set_title("(c) ln T at W=%g eV, L=%g nm" % (Wmax, Lmax))
    fig.tight_layout()
    return [fig]


if __name__ == "__main__":
    _runner.run_one(__import__(__name__))
