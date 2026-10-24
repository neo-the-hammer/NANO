"""C4 -- the concurrent execution modes give the same answers.

The proposed method runs the CPU and the GPU at the same time (see
src/vides_rgf_batch.h, "Concurrency"): CPU threads over energies, one
batch split between the GPU and the CPU threads, and the eigen
self-energies of the next chunk computed while the GPU solves the current
one.  This checks every mode against the stock serial CPU path.

Native harness (the solver and the decimation alone, identical inputs):
  threads   CPU, all threads vs CPU, 1 thread.  Each energy runs the same
            scalar code into its own slot, so the expected difference is
            exactly 0, not merely small.
  hybrid    CPU+GPU split vs CPU, 1 thread: GPU accuracy (C1, C2).
Whole devices (C3's set, through the Python API), each mode in its own
process:
  cpuN      vs cpu1: exactly 0 expected (also the charge assembly is
            sequential in energy, as before)
  gpu       GPU only (as in the first CUDA version)
  hybrid    the proposed default: GPU + CPU concurrently, pipelined
Tolerances as in C1/C3: 1e-10 for the solver, 1e-7 for decimation and the
devices that use it, 1e-5 where CPU and GPU use different self-energy
methods (the Hamiltonian path at its default eta, as in C3 ham-default).
"""

from common import (Step, run_native, with_gpu, u_rgf, u_decim, device_step, rel_to_peak,
                    fmt_e, MODES, MODE_LABEL, CPU_C, GPU_C, ALT_C, MUTED)

ID = "C4"
TITLE = "Concurrent CPU+GPU modes: same answers as the serial CPU"
KIND = "correctness"

NATIVE = [("std", 8, 30), ("std", 24, 40), ("lake", 16, 30), ("mode", 12, 30)]
DECIM = [("gnr", 8), ("hsrc", 8)]
DEV_TOL = {"cnt": 1e-8, "cntmode": 1e-8, "gnr": 1e-7, "hamiltonian": 1e-5, "cnt_schottky": 1e-8,
           "vacancy": 1e-5, "snw": 1e-7, "graphene": 1e-7, "tmd": 1e-5}


def devices(quick):
    g = [-2.995, 2.995, 0.02] if quick else [-2.995, 2.995, 0.01]
    return [("cnt", {"kind": "cnt", "size": [10, 2.0], "grid": g}),
            ("cntmode", {"kind": "cntmode", "size": [10, 2.0], "grid": g, "nmodes": 4}),
            ("gnr", {"kind": "gnr", "size": [6, 2.0], "grid": g}),
            ("hamiltonian", {"kind": "hamiltonian", "size": [5, 12], "grid": g}),
            ("cnt_schottky", {"kind": "cnt_schottky", "size": [10, 2.0], "grid": g}),
            ("vacancy", {"kind": "vacancy", "size": [0, 0], "grid": [-1.4975, 1.4975, 0.01]}),
            ("snw", {"kind": "snw", "size": [3.85, 15.0], "grid": [0.05, 9.95, 0.1]}),
            ("graphene", {"kind": "graphene", "size": [8.0, 6], "grid": [-0.4975, 0.4975, 0.01]}),
            ("tmd", {"kind": "tmd", "size": [8.0, 6], "grid": [-1.4975, 1.4975, 0.01]})]


def steps(ctx):
    out, g, NB = [], ctx["gpu"], 64 if ctx["quick"] else 128
    for var, n, Nc in NATIVE:
        a = ["modes", var, n, Nc, NB, 1e-5, 1]
        out.append(Step("native %s n=%d Nc=%d" % (var, n, Nc), "rgf", with_gpu(u_rgf(n, Nc, NB), 6, g),
                        (lambda a=a: dict(run_native(ctx["exe"], a), part="rgf")), g))
    for cell, n in DECIM:
        a = ["dmodes", cell, n, 48, 1e-5, 1]
        out.append(Step("native decimation %s n=%d" % (cell, n), "decim",
                        with_gpu(u_decim(n, 48), 6, g),
                        (lambda a=a: dict(run_native(ctx["exe"], a), part="decim")), g))
    for name, spec in devices(ctx["quick"]):
        for mode in ("cpu1", "cpuN", "gpu", "hybrid"):
            gpu, env = MODES[mode]
            out.append(device_step("%s %s" % (name, mode), spec, gpu, None, 1,
                                   {"device": name, "mode": mode}, env))
    return out


def analyze(recs, ctx):
    rows, verdicts, worst_thr = [], [], 0.0
    gpu_any = False
    for r in recs:
        if "error" in r or r.get("part") not in ("rgf", "decim"):
            continue
        if r["part"] == "rgf":
            thr = r["threads_maxabs"]
            hyb = max(r["hyb_A1"]["relpeak"], r["hyb_A2"]["relpeak"], r["hyb_T"]["relpeak"]) if r["gpu"] else None
            tol, what = 1e-10, "%s n=%d Nc=%d" % (r["variant"], r["n"], r["Nc"])
        else:
            thr, hyb = r["threads_maxdiff"], (r["hyb_rel_diff"] if r["gpu"] else None)
            tol, what = 1e-7, "decimation %s n=%d" % (r["cell"], r["n"])
        gpu_any = gpu_any or r["gpu"]
        worst_thr = max(worst_thr, thr)
        v = "PASS" if thr == 0 and (hyb is None or hyb < tol) else "FAIL"
        if hyb is not None and thr == 0 and tol <= hyb < 100 * tol:
            v = "WARN"
        verdicts.append(v)
        if r["part"] == "rgf":
            gonly = max(r["gpu_A1"]["relpeak"], r["gpu_T"]["relpeak"]) if r["gpu"] else None
        else:
            gonly = r["gpu_rel_diff"] if r["gpu"] else None
        rows.append(["native", what, "%d" % r["threads"], "%.0e" % thr if thr else "0 (exact)",
                     fmt_e(gonly) if gonly is not None else "-", fmt_e(hyb) if hyb is not None else "-",
                     "%.0f%%" % (100 * r["cpu_share"]) if r.get("cpu_share", -1) >= 0 else "-", "%g" % tol, v])
    by = {}
    for r in recs:
        if "error" not in r and "mode" in r:
            by.setdefault(r["device"], {})[r["mode"]] = r
    for name, m in by.items():
        if "cpu1" not in m:
            continue
        ref, tol = m["cpu1"], DEV_TOL[name]
        d = {}
        for mode in ("cpuN", "gpu", "hybrid"):
            if mode in m:
                x = m[mode]
                d[mode] = max(rel_to_peak(ref["T"], x["T"]),
                              rel_to_peak(ref["charge"], x["charge"]) if max(map(abs, ref["charge"])) > 1e-3
                              else max(abs(a - b) for a, b in zip(ref["charge"], x["charge"])) / 1e-3,
                              abs(ref["current"] - x["current"]) / max(abs(ref["current"]), 1e-300))
        gpu_used = m.get("hybrid", {}).get("gpu_used", False)
        gpu_any = gpu_any or gpu_used
        worst_thr = max(worst_thr, d.get("cpuN", 0))
        v = "PASS"
        if d.get("cpuN", 0) != 0:
            v = "FAIL"
        for mode in ("gpu", "hybrid"):
            if mode in d and gpu_used and d[mode] >= tol:
                v = "FAIL" if d[mode] >= 100 * tol else ("WARN" if v == "PASS" else v)
        verdicts.append(v)
        rows.append(["device", name, "-", "%.0e" % d["cpuN"] if d.get("cpuN") else "0 (exact)",
                     fmt_e(d.get("gpu")), fmt_e(d.get("hybrid")), "-", "%g" % tol, v])
    n_err = sum(1 for r in recs if "error" in r)
    if n_err:
        verdict, head = "FAIL", "%d run(s) errored" % n_err
    elif not gpu_any:
        verdict = "FAIL" if "FAIL" in verdicts else "NO GPU"
        head = "threaded CPU identical to serial (max diff %g); no GPU for the hybrid checks" % worst_thr
    else:
        verdict = "FAIL" if "FAIL" in verdicts else ("WARN" if "WARN" in verdicts else "PASS")
        head = ("threaded CPU bit-identical to serial (max diff %g); hybrid and GPU-only within "
                "tolerance in %d/%d cases" % (worst_thr, verdicts.count("PASS"), len(verdicts)))
    return {"verdict": verdict, "headline": head,
            "table": {"cols": ["level", "case", "threads", "CPU threads vs 1", "GPU only vs 1",
                               "hybrid vs 1", "CPU share", "tol", "verdict"], "rows": rows}}


def plot(recs, summ, ctx, plt):
    by = {}
    for r in recs:
        if "error" not in r and "mode" in r:
            by.setdefault(r["device"], {})[r["mode"]] = r
    P = [(k, v) for k, v in by.items() if "cpu1" in v]
    if not P:
        return []
    fig, ax = plt.subplots(1, len(P), figsize=(4 * len(P), 3.4), squeeze=False)
    fig.suptitle("C4  Transmission in every execution mode (lines must coincide)", x=0.01, ha="left")
    cols = {"cpu1": CPU_C, "cpuN": ALT_C[2], "gpu": ALT_C[3], "hybrid": GPU_C}
    styles = {"cpu1": "-", "cpuN": "--", "gpu": ":", "hybrid": "-."}
    for i, (name, m) in enumerate(P):
        a = ax[0][i]
        for mode in ("cpu1", "cpuN", "gpu", "hybrid"):
            if mode in m:
                a.plot(m[mode]["E"], m[mode]["T"], styles[mode], color=cols[mode],
                       lw=2.2 if mode == "cpu1" else 1.1, label=MODE_LABEL[mode])
        a.set_title(name)
        a.set_xlabel("E (eV)")
        if i == 0:
            a.set_ylabel("T(E)")
            a.legend(fontsize=6.5)
    fig.tight_layout()
    return [fig]
