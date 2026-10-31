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
  ham-default  as above with each backend's default self-energy -- what a
               user gets.  Since the eigen method became the default on GPU
               runs too (pipelined with the GPU solve), both use it; the
               tolerance stays 1e-5 so older builds, where the GPU default
               was decimation, are judged the same way.

  cnt-schottky, gnr-schottky   Schottky contacts (closed-form self-energy)
  gnr-fast     nanoribbon_fast_ohmic: a ribbon built in Python, H path
  vacancy      ribbon with one atom removed, from test/ribbon5_hole.xyz
  si-nanowire  sp3d5s* silicon nanowire, 10 orbitals per atom (H path)
  graphene-2d, bilayer-2d, tmd-2d   2D flakes summed over the transverse
               wave vector ky: one H_charge_T per ky

Tolerances (max diff relative to peak, over T, charge and current):
  cnt, cntmode          1e-8   same arithmetic, different summation order
  gnr, hamiltonian      1e-7   plus GPU vs CPU decimation, which agree only
                               to the decimation's own accuracy (see C2)
  ham-default           1e-5   eigen vs decimation (see A1, A2)
  schottky              1e-8;  si-nanowire, graphene-2d 1e-7
                               (eta < 1e-6: eigen on both)
  gnr-fast              1e-6   eta = 1e-8 (the class default) through the
                               Lake recursion: conditioning ~ 1/eta, the
                               T4 run gave 1.0e-7 in charge, T 4e-12
  vacancy, bilayer-2d, tmd-2d   1e-5 (eta = 1e-5; eigen vs decimation on
                               builds older than the eigen default)
WARN up to 100x the tolerance, FAIL beyond.

Grids are offset by half a step so no energy lands exactly on a zigzag
edge-state resonance (see test/test_gpu_vs_cpu.py, case hamiltonian).
"""

from common import device_step, rel_to_peak, fmt_e, positive, CPU_C, GPU_C, MUTED

ID = "C3"
TITLE = "Whole devices end to end: CPU vs GPU"
KIND = "correctness"
TOLS = {"cnt": 1e-8, "cntmode": 1e-8, "gnr": 1e-7, "hamiltonian": 1e-7,
        "ham-default": 1e-5, "cnt-schottky": 1e-8, "gnr-schottky": 1e-8, "gnr-fast": 1e-6,
        "vacancy": 1e-5, "si-nanowire": 1e-7, "graphene-2d": 1e-7, "bilayer-2d": 1e-5,
        "tmd-2d": 1e-5}


def devices(quick):
    g = [-2.995, 2.995, 0.02] if quick else [-2.995, 2.995, 0.01]
    return [
        ("cnt", {"kind": "cnt", "size": [10, 2.0], "grid": g}, None),
        ("cntmode", {"kind": "cntmode", "size": [10, 2.0], "grid": g, "nmodes": 4}, None),
        ("gnr", {"kind": "gnr", "size": [6, 2.0], "grid": g}, None),
        ("hamiltonian", {"kind": "hamiltonian", "size": [5, 12], "grid": g}, "dec"),
        ("ham-default", {"kind": "hamiltonian", "size": [5, 12], "grid": g}, None),
    ] + more_devices(quick)


def more_devices(quick):
    """The other device classes of the library (see worker.py)."""
    g = [-1.4975, 1.4975, 0.01] if quick else [-1.4975, 1.4975, 0.005]
    gk = [-0.4975, 0.4975, 0.01] if quick else [-0.4975, 0.4975, 0.005]
    nk = 6 if quick else 16
    return [
        ("cnt-schottky", {"kind": "cnt_schottky", "size": [10, 2.0], "grid": g}, None),
        ("gnr-schottky", {"kind": "gnr_schottky", "size": [6, 2.0], "grid": g}, None),
        ("gnr-fast", {"kind": "gnr_fast", "size": [4, 2.0], "grid": g}, None),
        ("vacancy", {"kind": "vacancy", "size": [0, 0], "grid": g}, None),
        ("si-nanowire", {"kind": "snw", "size": [3.85, 15.0], "grid": [0.05, 9.95, 0.1 if quick else 0.05]}, None),
        ("graphene-2d", {"kind": "graphene", "size": [8.0 if quick else 15.0, nk], "grid": gk}, None),
        ("bilayer-2d", {"kind": "bilayer", "size": [8.0 if quick else 10.0, nk], "grid": gk}, None),
        ("tmd-2d", {"kind": "tmd", "size": [8.0 if quick else 15.0, nk], "grid": g}, None),
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
