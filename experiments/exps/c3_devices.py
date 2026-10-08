"""C3 -- whole devices end to end, CPU vs GPU.

Each device is solved through the public Python API (charge_T / current)
twice, in fresh processes with VIDES_GPU=0 and VIDES_GPU=1.  This covers
every GPU block in context: batched self-energies, batched RGF, and the
charge/transmission assembly in the four entry points.

  cnt          CNT real space    CNT_charge_T      RGF std
  cntmode      CNT mode space    CNTmode_charge_T  RGF mode
  gnr          graphene ribbon   GNR_charge_T      RGF std + GPU decimation
  hamiltonian  generic H         H_charge_T        RGF Lake + GPU decimation
               (both runs pinned to VIDES_SELFH=dec, so any diff is the GPU's)
  ham-default  as above with each backend's default self-energy: the eigen
               method on the CPU, decimation on the GPU -- what a user gets.
               Differences here are the methods', not the GPU's.

Tolerances (max diff relative to peak, over T, charge and current):
  cnt, cntmode          1e-8   same arithmetic, different summation order
  gnr, hamiltonian      1e-7   plus GPU vs CPU decimation, which agree only
                               to the decimation's own accuracy (see C2)
  ham-default           1e-5   eigen vs decimation (see A1, A2)
WARN up to 100x the tolerance, FAIL beyond.

Grids are offset by half a step so no energy lands exactly on a zigzag
edge-state resonance (see test/test_gpu_vs_cpu.py, case hamiltonian).
"""

from common import device_step, rel_to_peak, fmt_e, positive, CPU_C, GPU_C, MUTED

ID = "C3"
TITLE = "Whole devices end to end: CPU vs GPU"
KIND = "correctness"
TOLS = {"cnt": 1e-8, "cntmode": 1e-8, "gnr": 1e-7, "hamiltonian": 1e-7,
        "ham-default": 1e-5}


def devices(quick):
    g = [-2.995, 2.995, 0.02] if quick else [-2.995, 2.995, 0.01]
    return [
        ("cnt", {"kind": "cnt", "size": [10, 2.0], "grid": g}, None),
        ("cntmode", {"kind": "cntmode", "size": [10, 2.0], "grid": g, "nmodes": 4}, None),
        ("gnr", {"kind": "gnr", "size": [6, 2.0], "grid": g}, None),
        ("hamiltonian", {"kind": "hamiltonian", "size": [5, 12], "grid": g}, "dec"),
        ("ham-default", {"kind": "hamiltonian", "size": [5, 12], "grid": g}, None),
    ]


def steps(ctx):
    out = []
    for name, spec, selfh in devices(ctx["quick"]):
        for gpu in (False, True):
            out.append(device_step("%s %s" % (name, "GPU" if gpu else "CPU"), spec, gpu,
                                   selfh, 1, {"device": name}))
    return out


def pairs(recs):
    by = {}
    for r in recs:
        if "error" in r:
            continue
        by.setdefault(r["device"], {})["gpu" if r["gpu_requested"] else "cpu"] = r
    return [(k, v["cpu"], v["gpu"]) for k, v in by.items() if "cpu" in v and "gpu" in v]


def metrics(c, g):
    return {"T": rel_to_peak(c["T"], g["T"]),
            "charge": rel_to_peak(c["charge"], g["charge"]),
            "current": abs(c["current"] - g["current"]) / max(abs(c["current"]), 1e-300)}


def analyze(recs, ctx):
    rows, verdicts = [], []
    gpu_any = False
    for name, c, g in pairs(recs):
        m = metrics(c, g)
        tol = TOLS[name]
        w = max(m.values())
        if not g["gpu_used"]:
            v = "NO GPU"
        else:
            gpu_any = True
            v = "PASS" if w < tol else ("WARN" if w < 100 * tol else "FAIL")
        verdicts.append(v)
        rows.append([name, "%d x %d, NE %d" % (c["n"], c["Nc"], c["NE"]), fmt_e(m["T"]),
                     fmt_e(m["charge"]), fmt_e(m["current"]), "%g" % tol, "%.2f s" % c["times"][0],
                     "%.2f s" % g["times"][0], v])
    n_err = sum(1 for r in recs if "error" in r)
    if n_err:
        verdict, head = "FAIL", "%d run(s) errored" % n_err
    elif not gpu_any:
        verdict, head = "NO GPU", "GPU runs fell back to the CPU (no CUDA device or CPU-only build)"
    else:
        verdict = "FAIL" if "FAIL" in verdicts else ("WARN" if "WARN" in verdicts else "PASS")
        same = [max(metrics(c, g).values()) for n, c, g in pairs(recs) if n != "ham-default"]
        head = "worst rel. diff in T / charge / current: %s (same method on both)" % fmt_e(max(same or [0]))
    return {"verdict": verdict, "headline": head,
            "table": {"cols": ["device", "size", "T diff", "charge diff", "current diff",
                               "tol", "CPU", "GPU", "verdict"], "rows": rows}}


def plot(recs, summ, ctx, plt):
    P = pairs(recs)
    if not P:
        return []
    fig, ax = plt.subplots(len(P), 3, figsize=(13, 2.6 * len(P)), squeeze=False)
    fig.suptitle("C3  Whole devices: CPU (line) vs GPU (dashed / markers)", x=0.01, ha="left")
    for i, (name, c, g) in enumerate(P):
        a, b, d = ax[i]
        a.plot(c["E"], c["T"], color=CPU_C, lw=1.6, label="CPU")
        a.plot(g["E"], g["T"], color=GPU_C, lw=1.0, ls="--", label="GPU")
        a.set_ylabel("%s\nT(E)" % name)
        diff = [abs(x - y) for x, y in zip(c["T"], g["T"])]
        b.semilogy(c["E"], positive(diff), color=MUTED, lw=1)
        b.set_ylabel("|ΔT|")
        x = list(range(len(c["charge"])))
        d.plot(x, c["charge"], color=CPU_C, lw=1.4, label="CPU")
        d.plot(x, g["charge"], "o", ms=1.8, color=GPU_C, label="GPU")
        d.set_ylabel("charge / site")
        dd = d.twinx()
        dd.semilogy(x, positive([abs(p - q) for p, q in zip(c["charge"], g["charge"])]),
                    color=MUTED, lw=0.6, alpha=0.8)
        dd.set_ylabel("|Δcharge|", color=MUTED)
        dd.grid(False)
        if i == 0:
            a.set_title("transmission")
            b.set_title("transmission difference")
            d.set_title("charge per site (grey: |difference|)")
            a.legend(fontsize=7)
            d.legend(fontsize=7, loc="upper left")
        if i == len(P) - 1:
            a.set_xlabel("E (eV)")
            b.set_xlabel("E (eV)")
            d.set_xlabel("site index")
    fig.tight_layout()
    return [fig]
