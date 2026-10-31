#!/usr/bin/env python3
"""E3 -- scaling to large devices: how far each backend goes in a time budget.

Graphene nanoribbons of growing width n (atoms across; the block of the
recursive Green's function is n x n and the contact cell 4n x 4n), 10 nm
long, 601 energies.  Each width is solved on the CPU (all threads) and on
the GPU.  Once a backend needs more than the time budget for one solve
(30 s quick, 300 s full), larger widths are skipped on it: the result is
the largest device each backend can solve within the budget, and the
fitted growth of the solve time with n (t ~ n^p; the dense algebra is
O(n^3), so p near 3 means the work, not overheads, dominates).

The GPU keeps going after the CPU stops, and the ratio of the two
largest widths is the practical gain a user sees: a bigger device in the
same wall time, not only the same device faster.
"""

import math

import _runner  # noqa: F401
from common import Step, run_device, steady, u_device, fmt_t, CPU_C, GPU_C

ID = "E3"
TITLE = "Scaling to large devices: the largest ribbon each backend solves in a time budget"
KIND = "speed"


def plan(quick):
    if quick:
        return [6, 12, 18, 24], 10.0, 30.0
    return [6, 12, 18, 24, 32, 40, 48, 64], 10.0, 300.0


def steps(ctx):
    widths, L, budget = plan(ctx["quick"])
    g = [-1.5025, 1.5025, 0.005]
    state = {False: False, True: False}          # backend already over budget?
    out = []
    for n in widths:
        spec = {"kind": "gnr", "size": [n, L], "grid": g}
        for gpu in (False, True):
            def fn(spec=spec, gpu=gpu, n=n):
                if state[gpu]:
                    return {"n": n, "gpu_req": gpu, "skipped": True}
                r = run_device(spec, gpu, repeat=2 if gpu else 1)
                t = steady(r["times"])
                if t > budget:
                    state[gpu] = True
                r.update(n=n, gpu_req=gpu, t=t, budget=budget)
                return r
            out.append(Step("gnr n=%d %s" % (n, "GPU" if gpu else "CPU"), "dev_gnr_%s" % ("gpu" if gpu else "cpu"),
                            u_device(spec, gpu, None, 2 if gpu else 1), fn, gpu))
    return out


def fit_exponent(pts):
    pts = [(n, t) for n, t in pts if t > 0]
    if len(pts) < 3:
        return None
    pts = pts[len(pts) // 2:] if len(pts) > 4 else pts      # large n: past the fixed costs
    xs = [math.log(n) for n, _ in pts]
    ys = [math.log(t) for _, t in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else None


def series(recs, gpu):
    return sorted((r["n"], r["t"]) for r in recs
                  if "error" not in r and r.get("gpu_req") == gpu and not r.get("skipped") and "t" in r)


def analyze(recs, ctx):
    widths, L, budget = plan(ctx["quick"])
    sc, sg = series(recs, False), series(recs, True)
    tc, tg = dict(sc), dict(sg)
    gpu = any(r.get("gpu_used") for r in recs if r.get("gpu_req"))
    rows = []
    for n in widths:
        rows.append(["%d" % n, "%d" % (4 * n), fmt_t(tc.get(n)).strip() if n in tc else "skipped",
                     fmt_t(tg.get(n)).strip() if n in tg else "skipped",
                     "%.1fx" % (tc[n] / tg[n]) if n in tc and n in tg else "-"])
    within = lambda s: max([n for n, t in s if t <= budget] or [0])
    nc, ng = within(sc), within(sg)
    pc, pg = fit_exponent(sc), fit_exponent(sg)
    n_err = sum(1 for r in recs if "error" in r)
    if n_err:
        verdict, head = "FAIL", "%d run(s) errored" % n_err
    elif not gpu:
        verdict, head = "NO GPU", "CPU only: largest width within %s: %d" % (fmt_t(budget).strip(), nc)
    else:
        verdict = "INFO"
        head = ("within %s per solve: CPU up to n=%d, GPU up to n=%d%s; time ~ n^%s (CPU), n^%s (GPU)"
                % (fmt_t(budget).strip(), nc, ng,
                   " (the largest width tried)" if ng == widths[-1] else "",
                   "%.1f" % pc if pc else "?", "%.1f" % pg if pg else "?"))
    return {"verdict": verdict, "headline": head,
            "table": {"cols": ["width n", "contact cell", "CPU per solve", "GPU per solve", "speedup"],
                      "rows": rows}}


def plot(recs, summ, ctx, plt):
    sc, sg = series(recs, False), series(recs, True)
    if not sc and not sg:
        return []
    budget = plan(ctx["quick"])[2]
    fig, a = plt.subplots(figsize=(7, 4))
    fig.suptitle("E3  Solve time against ribbon width (10 nm, 601 energies)", x=0.01, ha="left")
    if sc:
        a.loglog([n for n, _ in sc], [t for _, t in sc], "o-", color=CPU_C, label="CPU, all threads")
    if sg:
        a.loglog([n for n, _ in sg], [t for _, t in sg], "s-", color=GPU_C, label="GPU")
    a.axhline(budget, color="#898781", ls="--", lw=1)
    a.annotate("time budget", (a.get_xlim()[0], budget), xytext=(4, 4), textcoords="offset points",
               fontsize=8, color="#898781")
    a.set_xlabel("ribbon width n (atoms)")
    a.set_ylabel("seconds per solve")
    a.legend()
    fig.tight_layout()
    return [fig]


if __name__ == "__main__":
    _runner.run_one(__import__(__name__))
