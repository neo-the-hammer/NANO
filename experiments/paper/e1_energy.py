#!/usr/bin/env python3
"""E1 -- energy to solution: joules per NEGF solve on the CPU and on the GPU.

Speed alone does not settle whether a GPU pays off; energy per result
does, and reviewers of HPC software ask for it.  Each device is solved on
the CPU (all threads) and on the GPU (the default hybrid method) while
power.py samples the GPU board power (nvidia-smi, every 100 ms) and,
where the machine exposes them, the CPU package energy counters (RAPL).
An idle measurement first gives the floor that both runs pay anyway.

Energy per solve = mean power during the run x the steady solve time
(the best repeat after the first, which carries start-up).  System power
is CPU package + every GPU board: a CPU run still powers an idle GPU,
which is reported separately as "above idle".  Without RAPL (most cloud
VMs) the CPU part is unavailable and only GPU board energy is given; the
verdict then says so instead of guessing.
"""

import _runner  # noqa: F401  (puts experiments/ and paper/ on sys.path)
import power
from common import Step, run_device, steady, u_device, fmt_t, CPU_C, GPU_C, MUTED

ID = "E1"
TITLE = "Energy to solution: joules per NEGF solve, CPU vs GPU"
KIND = "speed"


def devices(quick):
    g = [-1.5025, 1.5025, 0.01] if quick else [-1.5025, 1.5025, 0.005]
    if quick:
        return [("cnt 13x5", {"kind": "cnt", "size": [13, 5.0], "grid": g}),
                ("gnr 8x5", {"kind": "gnr", "size": [8, 5.0], "grid": g}),
                ("H 8x24", {"kind": "hamiltonian", "size": [8, 24], "grid": g})]
    return [("cnt 19x10", {"kind": "cnt", "size": [19, 10.0], "grid": g}),
            ("gnr 12x10", {"kind": "gnr", "size": [12, 10.0], "grid": g}),
            ("H 16x40", {"kind": "hamiltonian", "size": [16, 40], "grid": g}),
            ("graphene 2D 20nm x32 ky", {"kind": "graphene", "size": [20.0, 32],
                                         "grid": [-0.4975, 0.4975, 0.005]})]


def steps(ctx):
    # cost class "sim" counts nominal seconds x 1e9 (see run_all.calibrate)
    out = [Step("idle power (3 s)", "sim", 3e9, lambda: dict(power.idle(3.0), idle=True), False)]
    for name, spec in devices(ctx["quick"]):
        for gpu in (False, True):
            def fn(spec=spec, gpu=gpu, name=name):
                s = power.PowerSampler().start()
                r = run_device(spec, gpu, repeat=3)
                p = s.stop()
                r.update(device=name, gpu_req=gpu, power=p)
                return r
            out.append(Step("%s %s" % (name, "GPU" if gpu else "CPU"), "dev_%s_%s" % (spec["kind"], "gpu" if gpu else "cpu"),
                            u_device(spec, gpu, None, 3), fn, gpu))
    return out


def _sys_w(p, idle_gpu):
    """System power during a run: CPU package (if known) + GPU boards."""
    g = p.get("gpu_w_mean")
    c = p.get("cpu_w_mean")
    if g is None and c is None:
        return None
    return (c or 0.0) + (g if g is not None else (idle_gpu or 0.0))


def analyze(recs, ctx):
    idle = next((r for r in recs if r.get("idle")), {})
    idle_gpu, idle_cpu = idle.get("gpu_w_mean"), idle.get("cpu_w_mean")
    by = {}
    for r in recs:
        if "error" in r or "device" not in r:
            continue
        by.setdefault(r["device"], {})["gpu" if r["gpu_req"] else "cpu"] = r
    rows, ratios, rapl = [], [], idle_cpu is not None
    for name, m in by.items():
        if "cpu" not in m or "gpu" not in m:
            continue
        tc, tg = steady(m["cpu"]["times"]), steady(m["gpu"]["times"])
        wc, wg = _sys_w(m["cpu"]["power"], idle_gpu), _sys_w(m["gpu"]["power"], idle_gpu)
        ec = wc * tc if wc is not None else None
        eg = wg * tg if wg is not None else None
        # above idle: what the solve itself costs on top of a powered-on node
        floor = (idle_cpu or 0.0) + (idle_gpu or 0.0)
        ac = (wc - floor) * tc if wc is not None else None
        ag = (wg - floor) * tg if wg is not None else None
        if ec and eg:
            ratios.append((ec / eg, name))
        f = lambda x, u: ("%.3g %s" % (x, u)) if x is not None else "n/a"
        rows.append([name, fmt_t(tc).strip(), fmt_t(tg).strip(), "%.1fx" % (tc / tg),
                     f(m["cpu"]["power"].get("cpu_w_mean"), "W"), f(m["gpu"]["power"].get("gpu_w_mean"), "W"),
                     f(ec, "J"), f(eg, "J"), f(ac, "J"), f(ag, "J"),
                     "%.2fx" % (ec / eg) if ec and eg else "n/a"])
    n_err = sum(1 for r in recs if "error" in r)
    if n_err:
        verdict, head = "FAIL", "%d run(s) errored" % n_err
    elif idle_gpu is None:
        verdict, head = "NO GPU", "no GPU power readings (nvidia-smi unavailable)"
    elif not ratios:
        verdict, head = "INFO", "no device pair finished"
    else:
        b = max(ratios)
        verdict = "INFO"
        head = ("GPU uses %.1fx less energy per solve at best (%s)%s"
                % (b[0], b[1], "" if rapl else "; CPU energy counters (RAPL) not exposed: "
                   "system energy counts the GPU boards only"))
    notes = ["idle: GPU boards %s W, CPU package %s W" % (
        "%.1f" % idle_gpu if idle_gpu is not None else "n/a",
        "%.1f" % idle_cpu if idle_cpu is not None else "n/a (no RAPL)")]
    return {"verdict": verdict, "headline": head, "notes": notes,
            "table": {"cols": ["device", "CPU time", "GPU time", "speedup", "CPU pkg W (CPU run)",
                               "GPU W (GPU run)", "J/solve CPU run", "J/solve GPU run",
                               "above idle, CPU run", "above idle, GPU run", "energy ratio"],
                      "rows": rows}}


def plot(recs, summ, ctx, plt):
    rows = (summ.get("table") or {}).get("rows") or []
    pts = [(r[0], float(r[6].split()[0]), float(r[7].split()[0])) for r in rows
           if r[6] != "n/a" and r[7] != "n/a"]
    if not pts:
        return []
    fig, a = plt.subplots(figsize=(8, 3.6))
    fig.suptitle("E1  Energy per NEGF solve (system: CPU package + GPU boards)", x=0.01, ha="left")
    x = list(range(len(pts)))
    a.bar([i - 0.2 for i in x], [p[1] for p in pts], 0.4, color=CPU_C, label="CPU run")
    a.bar([i + 0.2 for i in x], [p[2] for p in pts], 0.4, color=GPU_C, label="GPU run")
    a.set_xticks(x)
    a.set_xticklabels([p[0] for p in pts], fontsize=8)
    a.set_yscale("log")
    a.set_ylabel("joules per solve")
    a.legend()
    for i, p in enumerate(pts):
        a.annotate("%.1fx" % (p[1] / p[2]), (i, max(p[1], p[2])), ha="center", va="bottom",
                   fontsize=8, color=MUTED, xytext=(0, 3), textcoords="offset points")
    fig.tight_layout()
    return [fig]


if __name__ == "__main__":
    _runner.run_one(__import__(__name__))
