"""S7 -- whole devices in every execution mode: the proposed method end to end.

Each device runs through the Python API in four processes, one per mode
(common.MODES):

  cpu1    CPU, 1 thread: the stock code path, the baseline
  cpuN    CPU, all threads
  gpu     GPU only, as in the first CUDA version (no CPU share, no
          pipeline, no memory pool)
  hybrid  the proposed default: GPU and CPU threads concurrently on every
          batch (adaptive split), eigen self-energies pipelined with the
          GPU solve, persistent GPU state

GPU modes solve twice and report the second solve (steady state, what each
further step of a self-consistent loop costs; it also lets the hybrid
split adapt).  Devices cover every path and the regimes where each side
wins: CNT (solver-bound), GNR (decimation-bound), Schottky contacts,
a ribbon with a vacancy, an sp3d5s* silicon nanowire (10 orbitals per
atom), 2D graphene, bilayer and TMD flakes summed over ky, the Hamiltonian path
with decimation (eta = 1e-5) and with the CPU-only eigen self-energy
(eta = 1e-8, where the pipeline matters), plus a structure summed over a
transverse wave vector (R2's graphene: 184 short solves, where the
per-call overhead of S6 matters).  "diff" is the worst relative
difference of T, charge and current from cpu1 (charge against
max(peak, 1e-3)); same tolerances as C4.
"""

from common import (Step, device_step, run_sim, steady, fmt_t, fmt_e, rel_to_peak, MODES, MODE_LABEL,
                    CPU_C, GPU_C, ALT_C, MUTED)

ID = "S7"
TITLE = "Whole devices in every execution mode (proposed vs stock)"
KIND = "speed"
ORDER = ("cpu1", "cpuN", "gpu", "hybrid")
COL = {"cpu1": "#9cb9de", "cpuN": CPU_C, "gpu": ALT_C[3], "hybrid": GPU_C}


def devices(quick):
    g = [-1.5025, 1.5025, 0.01] if quick else [-1.5025, 1.5025, 0.005]
    if quick:
        return [("cnt 13x5", {"kind": "cnt", "size": [13, 5.0], "grid": g}),
                ("gnr 8x5", {"kind": "gnr", "size": [8, 5.0], "grid": g}),
                ("H 8x24 eta1e-5", {"kind": "hamiltonian", "size": [8, 24], "grid": g, "eta": 1e-5}),
                ("H 8x24 eta1e-8", {"kind": "hamiltonian", "size": [8, 24], "grid": g, "eta": 1e-8}),
                ("Si nanowire 3.85x15A", {"kind": "snw", "size": [3.85, 15.0], "grid": [0.05, 9.95, 0.1]}),
                ("graphene 2D 10nm x8 ky", {"kind": "graphene", "size": [10.0, 8], "grid": [-0.4975, 0.4975, 0.01]}),
                ("TMD 2D 10nm x8 ky", {"kind": "tmd", "size": [10.0, 8], "grid": g})]
    return [("cnt 10x5", {"kind": "cnt", "size": [10, 5.0], "grid": g}),
            ("cnt 19x10", {"kind": "cnt", "size": [19, 10.0], "grid": g}),
            ("cntmode 19x10", {"kind": "cntmode", "size": [19, 10.0], "grid": g, "nmodes": 6}),
            ("gnr 6x5", {"kind": "gnr", "size": [6, 5.0], "grid": g}),
            ("gnr 12x10", {"kind": "gnr", "size": [12, 10.0], "grid": g}),
            ("H 8x40 eta1e-5", {"kind": "hamiltonian", "size": [8, 40], "grid": g, "eta": 1e-5}),
            ("H 16x40 eta1e-5", {"kind": "hamiltonian", "size": [16, 40], "grid": g, "eta": 1e-5}),
            ("H 16x40 eta1e-8", {"kind": "hamiltonian", "size": [16, 40], "grid": g, "eta": 1e-8}),
            ("CNT Schottky 19x10", {"kind": "cnt_schottky", "size": [19, 10.0], "grid": g}),
            ("GNR Schottky 12x10", {"kind": "gnr_schottky", "size": [12, 10.0], "grid": g}),
            ("vacancy ribbon", {"kind": "vacancy", "size": [0, 0], "grid": g}),
            ("Si nanowire 7.85x20A", {"kind": "snw", "size": [7.85, 20.0], "grid": [0.05, 9.95, 0.05]}),
            ("graphene 2D 20nm x32 ky", {"kind": "graphene", "size": [20.0, 32], "grid": [-0.4975, 0.4975, 0.005]}),
            ("bilayer 2D 15nm x32 ky", {"kind": "bilayer", "size": [15.0, 32], "grid": [-0.4975, 0.4975, 0.005]}),
            ("TMD 2D 20nm x32 ky", {"kind": "tmd", "size": [20.0, 32], "grid": g})]


def steps(ctx):
    out = []
    for name, spec in devices(ctx["quick"]):
        for mode in ORDER:
            gpu, env = MODES[mode]
            out.append(device_step("%s %s" % (name, mode), spec, gpu, None, 2 if gpu else 1,
                                   {"device": name, "mode": mode}, env))
    for mode in ORDER:
        gpu, env = MODES[mode]

        def fn(mode=mode, gpu=gpu, env=env):
            r = run_sim("reg_graphene", gpu, ctx["quick"], env=env)
            r.update(device="graphene k-sum (184 solves)", mode=mode, sim=True)
            return r
        out.append(Step("graphene k-sum %s" % mode, "sim_kgraphene_%s" % ("gpu" if gpu else "cpu"),
                        13e9 * (0.5 if gpu else 1), fn, gpu))
    return out


def t_of(r):
    return r["wall"] if r.get("sim") else steady(r["times"])


def diff(ref, x):
    if ref.get("sim"):
        return max(rel_to_peak(ref["series"][k], x["series"][k]) for k in ref["series"])
    c = ref["charge"]
    dc = max(abs(p - q) for p, q in zip(c, x["charge"])) / max(max(map(abs, c)), 1e-3)
    return max(rel_to_peak(ref["T"], x["T"]), dc,
               abs(ref["current"] - x["current"]) / max(abs(ref["current"]), 1e-300))


def table(recs):
    by, order = {}, []
    for r in recs:
        if "error" in r or "mode" not in r:
            continue
        if r["device"] not in by:
            order.append(r["device"])
        by.setdefault(r["device"], {})[r["mode"]] = r
    return [(d, by[d]) for d in order if "cpu1" in by[d]]


def analyze(recs, ctx):
    rows, sp_h, gain_h, bad, gpu = [], [], [], 0, False
    for name, m in table(recs):
        t = {k: t_of(v) for k, v in m.items()}
        d = {k: diff(m["cpu1"], v) for k, v in m.items() if k != "cpu1"}
        used = m.get("hybrid", {}).get("gpu_used", False)
        gpu = gpu or used
        if d.get("cpuN", 0) != 0:
            bad += 1
        if "hybrid" in t:
            sp_h.append((t["cpu1"] / t["hybrid"], name))
        if "hybrid" in t and "gpu" in t:
            gain_h.append((t["gpu"] / t["hybrid"], name))
        rows.append([name] + [fmt_t(t.get(k)).strip() for k in ORDER] +
                    ["%.1fx" % (t["cpu1"] / t[k]) if k in t else "-" for k in ("cpuN", "gpu", "hybrid")] +
                    ["%.2fx" % (t["gpu"] / t["hybrid"]) if "gpu" in t and "hybrid" in t else "-",
                     fmt_e(max(d.values())) if d else "-"])
    n_err = sum(1 for r in recs if "error" in r)
    if n_err:
        verdict, head = "FAIL", "%d run(s) errored" % n_err
    elif bad:
        verdict, head = "FAIL", "threaded CPU changed a result in %d device(s)" % bad
    elif not gpu:
        verdict, head = "NO GPU", "CPU modes only"
    else:
        b = max(sp_h)
        gm = sorted(x for x, _ in gain_h)
        verdict = "INFO"
        head = ("proposed vs stock CPU: up to %.1fx (%s); vs GPU-only: median %.2fx, up to %.2fx"
                % (b[0], b[1], gm[len(gm) // 2], gm[-1]))
    return {"verdict": verdict, "headline": head,
            "table": {"cols": ["device", "CPU 1 thr (stock)", "CPU all thr", "GPU only", "hybrid (proposed)",
                               "speedup cpuN", "speedup GPU", "speedup proposed", "proposed vs GPU only",
                               "max diff vs stock"], "rows": rows}}


def plot(recs, summ, ctx, plt):
    T = table(recs)
    if not T:
        return []
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [3, 2]})
    fig.suptitle("S7  Whole devices: time per solve in every execution mode", x=0.01, ha="left")
    a, b = ax
    w = 0.2
    for k, mode in enumerate(ORDER):
        xs, ys = [], []
        for i, (name, m) in enumerate(T):
            if mode in m:
                xs.append(i + (k - 1.5) * w)
                ys.append(t_of(m[mode]))
        a.bar(xs, ys, w, color=COL[mode], label=MODE_LABEL[mode])
    a.set_yscale("log")
    a.set_xticks(range(len(T)))
    a.set_xticklabels([t[0] for t in T], rotation=25, ha="right", fontsize=7.5)
    a.set_ylabel("seconds per solve")
    a.legend(fontsize=7)
    a.set_title("(a) time (log)")
    y = list(range(len(T)))
    for k, mode in enumerate(("cpuN", "gpu", "hybrid")):
        b.barh([v + (k - 1) * 0.27 for v in y],
               [t_of(m["cpu1"]) / t_of(m[mode]) if mode in m else 0 for _, m in T], 0.27,
               color=COL[mode], label=MODE_LABEL[mode])
    b.axvline(1, color="#d03b3b", lw=1, ls="--")
    b.set_yticks(y)
    b.set_yticklabels([t[0] for t in T], fontsize=7.5)
    b.invert_yaxis()
    b.set_xscale("log")
    b.set_xlabel("speedup over the stock CPU path")
    b.set_title("(b) speedup")
    b.legend(fontsize=6.5)
    fig.tight_layout()
    return [fig]
