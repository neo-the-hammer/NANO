#!/usr/bin/env python3
"""E4 -- numerical convergence and time to a given accuracy.

A faster solver is only useful at the accuracy a study needs, and the
two numerical knobs of a NEGF calculation are the energy step dE and the
broadening eta.  For a biased graphene nanoribbon (n = 8, 10 nm, linear
potential drop of 0.2 V, mu1 = 0, mu2 = -0.3 V) the current and the
charge are computed

  for dE = 40, 20, 10, 5, 2.5, 1.25 meV (eta = 1e-5), and
  for eta = 1e-3, 1e-4, 1e-5, 1e-6, 1e-8 (dE = 2.5 meV)

on the CPU and on the GPU.  References: for the dE sweep the CPU at the
finest dE, for the eta sweep the CPU at the smallest eta (same dE).  Results: the error of each run against that reference
(current: relative; charge: relative to max(peak, 1e-3)), the observed
order of convergence in dE, and the time each backend needs to reach a
1e-3 accuracy in current.  The two backends must converge to the same
limit: their curves overlap, and the GPU merely gets there sooner.
"""

import math

import _runner  # noqa: F401
from common import Step, run_device, steady, u_device, fmt_t, fmt_e, positive, CPU_C, GPU_C

ID = "E4"
TITLE = "Convergence in energy step and broadening; time to a given accuracy"
KIND = "accuracy"

DES = [0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125]
ETAS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-8]
TARGET = 1e-3


def spec(dE, eta, quick):
    lo, hi = -0.8, 0.5
    lo += 0.5 * dE                         # no energy exactly at a band centre
    n, L = (6, 6.0) if quick else (8, 10.0)
    return {"kind": "gnr", "size": [n, L], "grid": [lo, hi, dE], "phi": -0.02, "mu2": -0.3, "eta": eta}


def plan(quick):
    des = DES[:-1] if quick else DES
    return des, ETAS


def steps(ctx):
    q = ctx["quick"]
    des, etas = plan(q)
    out = []
    for gpu in (False, True):
        for dE in des:
            s = spec(dE, 1e-5, q)
            out.append(Step("dE=%g %s" % (dE, "GPU" if gpu else "CPU"), "dev_gnr_%s" % ("gpu" if gpu else "cpu"),
                            u_device(s, gpu, None, 2 if gpu else 1),
                            (lambda s=s, gpu=gpu, dE=dE: dict(run_device(s, gpu, repeat=2 if gpu else 1),
                                                              sweep="dE", dE=dE, eta=1e-5, gpu_req=gpu)), gpu))
        for eta in etas:
            if eta == 1e-5:
                continue                   # the dE sweep holds dE = 2.5 meV, eta = 1e-5
            s = spec(0.0025, eta, q)
            out.append(Step("eta=%g %s" % (eta, "GPU" if gpu else "CPU"), "dev_gnr_%s" % ("gpu" if gpu else "cpu"),
                            u_device(s, gpu, None, 2 if gpu else 1),
                            (lambda s=s, gpu=gpu, eta=eta: dict(run_device(s, gpu, repeat=2 if gpu else 1),
                                                                sweep="eta", dE=0.0025, eta=eta, gpu_req=gpu)),
                            gpu))
    return out


def _ok(r):
    return "error" not in r and "current" in r


def reference(recs, sweep="dE"):
    """dE sweep: the CPU at the finest dE.  eta sweep: the CPU at the
    smallest eta (same dE), so each sweep measures one knob only."""
    if sweep == "dE":
        cands = [r for r in recs if _ok(r) and not r["gpu_req"] and r["sweep"] == "dE"]
        return min(cands, key=lambda r: r["dE"]) if cands else None
    cands = [r for r in recs if _ok(r) and not r["gpu_req"] and r["dE"] == 0.0025]
    return min(cands, key=lambda r: r["eta"]) if cands else None


def err(r, ref):
    ei = abs(r["current"] - ref["current"]) / max(abs(ref["current"]), 1e-300)
    a, b = r["charge"], ref["charge"]
    if len(a) == len(b):
        ec = max(abs(x - y) for x, y in zip(a, b)) / max(max(map(abs, b)), 1e-3)
    else:
        ec = float("nan")
    return ei, ec


def analyze(recs, ctx):
    ref = reference(recs)
    gpu = any(r.get("gpu_used") for r in recs if r.get("gpu_req"))
    rows, tta = [], {}
    if ref:
        for r in sorted([r for r in recs if _ok(r)], key=lambda r: (r["sweep"], r["gpu_req"], -r["dE"], -r["eta"])):
            rr = ref if r["sweep"] == "dE" else (reference(recs, "eta") or ref)
            ei, ec = err(r, rr)
            t = steady(r["times"])
            rows.append([r["sweep"], "GPU" if r["gpu_req"] else "CPU", "%g" % (r["dE"] * 1e3), "%g" % r["eta"],
                         r["NE"], fmt_t(t).strip(), fmt_e(ei), fmt_e(ec)])
            if r["sweep"] == "dE" and ei <= TARGET and r is not ref:
                k = "GPU" if r["gpu_req"] else "CPU"
                tta[k] = min(tta.get(k, float("inf")), t)
    # observed order in dE: slope of log(err) against log(dE), CPU, excluding the reference
    order = None
    if ref:
        pts = [(r["dE"], err(r, ref)[0]) for r in recs
               if _ok(r) and r["sweep"] == "dE" and not r["gpu_req"] and r is not ref]
        pts = [(d, e) for d, e in pts if e > 0]
        if len(pts) >= 3:
            xs = [math.log(d) for d, _ in pts]
            ys = [math.log(e) for _, e in pts]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            order = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else None
    # both backends converge to the same limit: at the finest dE their currents agree
    same = None
    fin = {}
    for r in recs:
        if _ok(r) and r["sweep"] == "dE":
            k = r["gpu_req"]
            if k not in fin or r["dE"] < fin[k]["dE"]:
                fin[k] = r
    if False in fin and True in fin:
        same = abs(fin[True]["current"] - fin[False]["current"]) / max(abs(fin[False]["current"]), 1e-300)
    n_err = sum(1 for r in recs if "error" in r)
    if n_err:
        verdict, head = "FAIL", "%d run(s) errored" % n_err
    elif not ref:
        verdict, head = "FAIL", "no CPU reference"
    elif not gpu:
        verdict, head = "NO GPU", "CPU only; observed order in dE %s" % ("%.2f" % order if order else "?")
    else:
        verdict = "PASS" if same is not None and same < 1e-6 else "WARN"
        head = ("CPU and GPU converge to the same current (diff %s at the finest dE); order in dE %s; "
                "time to %g accuracy: CPU %s, GPU %s"
                % (fmt_e(same), "%.2f" % order if order else "?", TARGET,
                   fmt_t(tta.get("CPU")).strip() if "CPU" in tta else "not reached",
                   fmt_t(tta.get("GPU")).strip() if "GPU" in tta else "not reached"))
    return {"verdict": verdict, "headline": head,
            "table": {"cols": ["sweep", "backend", "dE (meV)", "eta", "energies", "time",
                               "current error", "charge error"], "rows": rows}}


def plot(recs, summ, ctx, plt):
    ref = reference(recs)
    ref_eta = reference(recs, "eta")
    if not ref:
        return []
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    fig.suptitle("E4  Convergence against the most refined CPU run", x=0.01, ha="left")
    for gpu, col, lab in ((False, CPU_C, "CPU"), (True, GPU_C, "GPU")):
        d = sorted([(r["dE"], err(r, ref)[0], steady(r["times"])) for r in recs
                    if _ok(r) and r["sweep"] == "dE" and r["gpu_req"] == gpu and r is not ref])
        e = sorted([(r["eta"], err(r, ref_eta)[0]) for r in recs
                    if ref_eta and _ok(r) and r["dE"] == 0.0025 and r["gpu_req"] == gpu and r is not ref_eta])
        if d:
            ax[0].loglog([x[0] * 1e3 for x in d], positive([x[1] for x in d]), "o-", color=col, label=lab)
            ax[2].loglog([x[2] for x in d], positive([x[1] for x in d]), "o-", color=col, label=lab)
        if e:
            ax[1].loglog([x[0] for x in e], positive([x[1] for x in e]), "o-", color=col, label=lab)
    ax[0].set_xlabel("energy step dE (meV)")
    ax[1].set_xlabel("broadening eta (eV)")
    ax[2].set_xlabel("seconds per solve")
    for a in ax:
        a.set_ylabel("relative error in current")
        a.legend()
    ax[2].axhline(TARGET, color="#898781", ls="--", lw=1)
    ax[0].set_title("(a) energy step")
    ax[1].set_title("(b) broadening")
    ax[2].set_title("(c) error against time")
    fig.tight_layout()
    return [fig]


if __name__ == "__main__":
    _runner.run_one(__import__(__name__))
