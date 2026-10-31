#!/usr/bin/env python3
"""E2 -- chunked energy batches: same answer, and what chunking costs.

The GPU path solves the energies in batches sized from free device memory
(vides_negf_chunk); a device too large for one batch is solved in several
chunks, and the eigen self-energies of chunk k+1 are computed while the
GPU solves chunk k (pipeline).  The main suite's devices fit in one batch,
so this path, which every large production run takes, was never
exercised there.  VIDES_GPU_BATCH caps the batch size to force it:

  uncapped (one batch) and caps of 256, 64 and 16 energies per batch

Each capped run must give the uncapped result (same arithmetic per
energy, so differences stay at round-off of the CPU/GPU split, tolerance
1e-8 relative to peak over T, charge and current), and every run must
agree with the CPU reference within the device tolerance of C3.  The
times show the cost of small batches: a cap of 16 leaves the GPU mostly
idle, which is why the code sizes batches from memory.
"""

import _runner  # noqa: F401
from common import Step, run_device, steady, rel_to_peak, u_device, fmt_t, fmt_e, CPU_C, GPU_C, ALT_C

ID = "E2"
TITLE = "Chunked energy batches: identical results, cost of small batches"
KIND = "correctness"

CAPS = [None, 256, 64, 16]
TOL = {"cnt": 1e-8, "gnr": 1e-7, "hamiltonian": 1e-7}


def devices(quick):
    g = [-1.5025, 1.5025, 0.01] if quick else [-1.5025, 1.5025, 0.0025]
    if quick:
        return [("cnt 13x5", {"kind": "cnt", "size": [13, 5.0], "grid": g}),
                ("H 8x24", {"kind": "hamiltonian", "size": [8, 24], "grid": g, "eta": 1e-8})]
    return [("cnt 19x10", {"kind": "cnt", "size": [19, 10.0], "grid": g}),
            ("gnr 12x10", {"kind": "gnr", "size": [12, 10.0], "grid": g}),
            ("H 16x40", {"kind": "hamiltonian", "size": [16, 40], "grid": g, "eta": 1e-8})]


def steps(ctx):
    out = []
    for name, spec in devices(ctx["quick"]):
        out.append(Step("%s CPU reference" % name, "dev_%s_cpu" % spec["kind"], u_device(spec, False),
                        (lambda spec=spec, name=name: dict(run_device(spec, False), device=name, cap="cpu"))))
        for cap in CAPS:
            env = {"VIDES_GPU_BATCH": str(cap)} if cap else {}
            out.append(Step("%s GPU batch %s" % (name, cap or "uncapped"), "dev_%s_gpu" % spec["kind"],
                            u_device(spec, True, None, 2) * (1 + (64.0 / cap if cap else 0)),
                            (lambda spec=spec, name=name, env=env, cap=cap:
                             dict(run_device(spec, True, repeat=2, env=env), device=name,
                                  cap=cap or "all")), True))
    return out


def _diff(a, b):
    c = a["charge"]
    dc = max(abs(p - q) for p, q in zip(c, b["charge"])) / max(max(map(abs, c)), 1e-3)
    return max(rel_to_peak(a["T"], b["T"]), dc,
               abs(a["current"] - b["current"]) / max(abs(a["current"]), 1e-300))


def analyze(recs, ctx):
    by = {}
    for r in recs:
        if "error" not in r and "device" in r:
            by.setdefault(r["device"], {})[r["cap"]] = r
    rows, verdicts, gpu = [], [], False
    for name, m in by.items():
        ref, full = m.get("cpu"), m.get("all")
        kind = (ref or full or {}).get("spec", {}).get("kind", "cnt")
        tol = TOL.get(kind, 1e-7)
        for cap in ["all"] + [c for c in CAPS if c]:
            if cap not in m:
                continue
            r = m[cap]
            gpu = gpu or r.get("gpu_used", False)
            d_full = _diff(full, r) if full and cap != "all" else 0.0
            d_cpu = _diff(ref, r) if ref else None
            ok = d_full < 1e-8 and (d_cpu is None or d_cpu < tol)
            v = "PASS" if ok else ("WARN" if d_full < 1e-6 and (d_cpu is None or d_cpu < 100 * tol) else "FAIL")
            verdicts.append(v)
            rows.append([name, cap, fmt_t(steady(r["times"])).strip(),
                         "%.2fx" % (steady(full["times"]) / steady(r["times"])) if full else "-",
                         fmt_e(d_full) if cap != "all" else "-", fmt_e(d_cpu) if d_cpu is not None else "-",
                         "%g" % tol, v])
    n_err = sum(1 for r in recs if "error" in r)
    if n_err:
        verdict, head = "FAIL", "%d run(s) errored" % n_err
    elif not gpu:
        verdict, head = "NO GPU", "GPU runs fell back to the CPU"
    else:
        verdict = "FAIL" if "FAIL" in verdicts else ("WARN" if "WARN" in verdicts else "PASS")
        head = "every batch size gives the one-batch result (%d/%d within tolerance)" % (
            verdicts.count("PASS"), len(verdicts))
    return {"verdict": verdict, "headline": head,
            "table": {"cols": ["device", "energies per batch", "GPU time", "speed vs one batch",
                               "diff vs one batch", "diff vs CPU", "tol vs CPU", "verdict"], "rows": rows}}


def plot(recs, summ, ctx, plt):
    by = {}
    for r in recs:
        if "error" not in r and "device" in r and r["cap"] != "cpu":
            by.setdefault(r["device"], []).append(r)
    if not by:
        return []
    fig, a = plt.subplots(figsize=(7, 3.6))
    fig.suptitle("E2  GPU time per solve against the batch size", x=0.01, ha="left")
    for k, (name, rs) in enumerate(by.items()):
        pts = sorted(((min(r["NE"], 1024) if r["cap"] == "all" else int(r["cap"]), steady(r["times"]))
                      for r in rs))
        a.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=ALT_C[k % len(ALT_C)], label=name)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("energies per GPU batch (largest: memory-sized, at most 1024)")
    a.set_ylabel("seconds per solve")
    a.legend()
    fig.tight_layout()
    return [fig]


if __name__ == "__main__":
    _runner.run_one(__import__(__name__))
