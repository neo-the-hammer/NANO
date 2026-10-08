#!/usr/bin/env python3
"""Run every CPU-vs-GPU experiment with one command, then show and compare.

    python3 experiments/run_all.py            # everything (minutes to tens of minutes)
    python3 experiments/run_all.py --quick    # small sizes, a few minutes
    python3 experiments/run_all.py --only C1 C2 S1
    python3 experiments/run_all.py --replot   # re-analyse saved results, no runs

In Colab / Jupyter, run it inside the kernel so figures appear inline:

    %run experiments/run_all.py --quick

Before running it prints the plan: every experiment with its estimated
time and the total ETA (from a ~5 s calibration on this machine).  While
running, each finished step prints its time, the experiment's remaining
time and the overall ETA; estimates are refined as steps finish.  After
each experiment its verdict, table and figure are shown; at the end a
summary table, and everything is written to experiments/results/
(JSON per experiment, PNG per figure, report.html with all of it).

Experiments (see each module in experiments/exps/ for the details):
  C1  NEGF solver (RGF) CPU vs GPU, all three variants        correctness
  C2  contact self-energy (decimation) CPU vs GPU             correctness
  C3  whole devices end to end (CNT, CNT mode, GNR, H)        correctness
  A1  self-energy accuracy vs eta, eigen vs decimation        accuracy
  A2  accuracy vs speed, backend x self-energy method         accuracy
  S1  NEGF solver speed: scaling in n, Nc, NB                 speed
  S2  self-energy speed: eigen vs decimation, CPU vs GPU      speed
  S3  whole-device speed with time breakdown                  speed

Prerequisite: the module built in src/ (./build.sh --gpu on a GPU
machine; ./build.sh gives a CPU-only build where GPU columns stay empty).
"""

import argparse
import base64
import html
import io
import json
import os
import platform
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "exps")):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as C  # noqa: E402
from common import Step  # noqa: E402

import c1_rgf, c2_selfenergy, c3_devices  # noqa: E402,E401
import a1_selfenergy_accuracy, a2_accuracy_speed  # noqa: E402,E401
import s1_rgf_scaling, s2_selfenergy_speed, s3_devices_speed  # noqa: E402,E401

EXPERIMENTS = [c1_rgf, c2_selfenergy, c3_devices, a1_selfenergy_accuracy,
               a2_accuracy_speed, s1_rgf_scaling, s2_selfenergy_speed,
               s3_devices_speed]
NEEDS_MODULE = {"C3", "A2", "S3"}


def print_table(cols, rows, out=sys.stdout, indent="   "):
    rows = [[str(c) for c in r] for r in rows]
    w = [max([len(str(c))] + [len(r[i]) for r in rows if i < len(r)]) for i, c in enumerate(cols)]
    out.write(indent + "  ".join(str(c).ljust(w[i]) for i, c in enumerate(cols)) + "\n")
    out.write(indent + "  ".join("-" * x for x in w) + "\n")
    for r in rows:
        out.write(indent + "  ".join(r[i].ljust(w[i]) for i in range(len(r))) + "\n")
    out.flush()


def calibrate(model, exe, gpu, want_devices, log):
    log("Calibrating the time estimates on this machine ...")
    t0 = time.time()
    C.run_native(exe, ["rgf", "std", 8, 16, 48, 1e-5, 1])
    dt = time.time() - t0
    # The CPU reference dominates this run; the GPU part (and its start-up)
    # is left to the per-class refinement.
    model.update("rgf", C.u_rgf(8, 16, 48), False, dt, also=("decim", "selfh", "_default"))
    # Device work is counted in the same units as the native processes;
    # GPU classes start from the assumed GPU/CPU ratio.
    model.scale["dev"] = model.scale["rgf"]
    model.scale["dev_gpu"] = model.scale["rgf"] * C.GPU_REL
    if want_devices:
        # A trivial device: its time is Python + module start-up, i.e. the
        # fixed cost of every device process.
        spec = {"kind": "gnr", "size": [3, 1.0], "grid": [-1.0, 1.0, 0.05]}
        t0 = time.time()
        C.run_device(spec, False)
        dt = time.time() - t0
        model.overhead["device"] = dt
        model.overhead["device_gpu"] = dt + 1.0     # + CUDA context

    log("  done in %s" % C.fmt_t(time.time() - t0 + dt).strip())


def fig_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


VERDICT_COLOR = {"PASS": "#2e8b57", "WARN": "#c9a227", "FAIL": "#d03b3b",
                 "INFO": "#2a78d6", "NO GPU": "#898781", "SKIPPED": "#898781",
                 "NOT RUN": "#898781"}


def write_report(path, header, results):
    h = []
    h.append("<!doctype html><html><head><meta charset='utf-8'>"
             "<meta name='viewport' content='width=device-width,initial-scale=1'>"
             "<title>ViDES CPU vs GPU</title><style>"
             ":root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9}"
             "@media (prefers-color-scheme: dark){:root{--bg:#161615;--ink:#f2f1ec;--ink2:#c3c2b7;"
             "--muted:#898781;--grid:#3a3936}}"
             "body{background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif;"
             "max-width:1200px;margin:0 auto;padding:16px}"
             "h1{font-size:22px;margin:8px 0}h2{font-size:17px;margin:28px 0 6px}"
             "p,li{color:var(--ink2)}table{border-collapse:collapse;margin:8px 0;font-size:12.5px;"
             "display:block;overflow-x:auto}td,th{border-bottom:1px solid var(--grid);padding:4px 10px;"
             "text-align:left;white-space:nowrap}th{color:var(--ink2);font-weight:600}"
             ".v{font-weight:700;color:#fff;border-radius:4px;padding:1px 7px;font-size:12px}"
             "img{max-width:100%;background:#fcfcfb;border-radius:6px;margin:6px 0}"
             "pre{white-space:pre-wrap;color:var(--ink2);font-size:12px}"
             "</style></head><body>")
    h.append("<h1>NanoTCAD ViDES — CPU vs GPU experiments</h1>")
    h.append("<p>%s</p>" % "<br>".join(html.escape(x) for x in header))
    h.append("<h2>Summary</h2><table><tr><th>id</th><th>experiment</th><th>verdict</th>"
             "<th>result</th><th>time</th><th>estimate</th></tr>")
    for r in results:
        col = VERDICT_COLOR.get(r["verdict"], "#898781")
        h.append("<tr><td><a href='#%s'>%s</a></td><td>%s</td><td><span class='v' style='background:%s'>"
                 "%s</span></td><td>%s</td><td>%s</td><td>%s</td></tr>"
                 % (r["id"], r["id"], html.escape(r["title"]), col, r["verdict"],
                    html.escape(r["headline"]), C.fmt_t(r.get("elapsed")).strip(),
                    C.fmt_t(r.get("estimate")).strip()))
    h.append("</table>")
    for r in results:
        h.append("<h2 id='%s'>%s — %s</h2>" % (r["id"], r["id"], html.escape(r["title"])))
        h.append("<p><b>%s</b> — %s</p>" % (r["verdict"], html.escape(r["headline"])))
        if r.get("doc"):
            h.append("<pre>%s</pre>" % html.escape(r["doc"].strip()))
        t = r.get("table")
        if t:
            h.append("<table><tr>%s</tr>" % "".join("<th>%s</th>" % html.escape(str(c)) for c in t["cols"]))
            for row in t["rows"]:
                h.append("<tr>%s</tr>" % "".join("<td>%s</td>" % html.escape(str(c)) for c in row))
            h.append("</table>")
        for b64 in r.get("figs", []):
            h.append("<img src='data:image/png;base64,%s'>" % b64)
        for e in r.get("errors", []):
            h.append("<pre>%s</pre>" % html.escape(e))
    h.append("</body></html>")
    with open(path, "w") as f:
        f.write("\n".join(h))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="small sizes (a few minutes)")
    ap.add_argument("--only", nargs="+", metavar="ID", help="run only these experiments")
    ap.add_argument("--skip", nargs="+", metavar="ID", default=[], help="skip these experiments")
    ap.add_argument("--replot", action="store_true",
                    help="no runs: re-analyse and re-plot the saved results")
    ap.add_argument("--no-plots", action="store_true", help="tables only")
    ap.add_argument("--list", action="store_true", help="list the experiments and exit")
    args = ap.parse_args(argv)
    out = sys.stdout

    def log(s=""):
        out.write(s + "\n")
        out.flush()

    if args.list:
        for e in EXPERIMENTS:
            log("  %-3s %-11s %s" % (e.ID, e.KIND, e.TITLE))
        return 0

    want = [x.upper() for x in args.only] if args.only else [e.ID for e in EXPERIMENTS]
    skip = {x.upper() for x in args.skip}
    exps = [e for e in EXPERIMENTS if e.ID in want and e.ID not in skip]
    if not exps:
        log("nothing to run (known ids: %s)" % " ".join(e.ID for e in EXPERIMENTS))
        return 2
    os.makedirs(C.RESULTS, exist_ok=True)
    nb = C.in_notebook()

    header = ["host: %s, %s, %d CPU threads, Python %s"
              % (platform.node(), platform.processor() or platform.machine(),
                 os.cpu_count() or 1, platform.python_version())]
    ctx = {"quick": args.quick}
    results = []

    if not args.replot:
        log("NanoTCAD ViDES -- CPU vs GPU experiments%s" % (" (quick)" if args.quick else ""))
        exe = C.build_native(log)
        info = C.run_native(exe, ["info"])
        ctx.update(exe=exe, gpu=bool(info["gpu"]))
        mod = C.module_path()
        header.append("GPU backend: %s" % info["backend"])
        header.append("Python module: %s%s" % (os.path.relpath(mod, C.ROOT) if mod else "not built",
                                                " (CUDA build)" if C.module_has_gpu() else
                                                (" (CPU-only build)" if mod else "")))
        for line in header:
            log("  " + line)
        if not ctx["gpu"]:
            log("  WARNING: no usable GPU -- GPU columns will be empty and correctness "
                "experiments report NO GPU.")
        skipped = []
        if not mod:
            skipped = [e for e in exps if e.ID in NEEDS_MODULE]
            if skipped:
                log("  WARNING: src/NanoTCAD_ViDESmod.so not built; skipping %s"
                    % " ".join(e.ID for e in skipped))
        run = [e for e in exps if e not in skipped]

        model = C.CostModel()
        calibrate(model, exe, ctx["gpu"], any(e.ID in NEEDS_MODULE for e in run), log)
        plan = [(e, e.steps(ctx)) for e in run]
        prog = C.Progress(plan, model, out)
        prog.plan()
        prog.start()
        t_all = time.time()
        interrupted = False
        for e, steps in plan:
            if interrupted:
                break
            prog.exp_start(e, steps)
            est = prog.remaining(steps)
            t_e = time.time()
            recs = []
            for i, s in enumerate(steps):
                s.est_at_start = s.est
                t0 = time.time()
                ok = True
                try:
                    rec = s.fn()
                except KeyboardInterrupt:
                    log("   interrupted -- analysing what has finished")
                    interrupted = True
                    break
                except Exception as ex:  # keep going: one failure must not lose the rest
                    ok = False
                    rec = {"error": "%s: %s" % (type(ex).__name__, str(ex)[-1500:])}
                s.actual = time.time() - t0
                rec["label"] = s.label
                rec["_t"] = s.actual
                recs.append(rec)
                prog.step_done(e, steps, i, ok)
            C.save_json("%s.json" % e.ID, {"id": e.ID, "title": e.TITLE, "quick": args.quick,
                                            "header": header, "elapsed": time.time() - t_e,
                                            "estimate": est, "records": recs})
            results.append(report_one(e, recs, ctx, args, nb, log, time.time() - t_e, est))
        for e in skipped:
            results.append({"id": e.ID, "title": e.TITLE, "verdict": "SKIPPED",
                            "headline": "module not built", "doc": e.__doc__})
        log("\nAll experiments finished in %s." % C.fmt_t(time.time() - t_all).strip())
    else:
        for e in exps:
            try:
                d = C.load_json("%s.json" % e.ID)
            except OSError:
                results.append({"id": e.ID, "title": e.TITLE, "verdict": "NOT RUN",
                                "headline": "no saved results", "doc": e.__doc__})
                continue
            ctx["quick"] = d.get("quick", False)
            header = d.get("header", header)
            results.append(report_one(e, d["records"], ctx, args, nb, log,
                                      d.get("elapsed"), d.get("estimate")))

    log("\n" + "=" * 78)
    log("SUMMARY")
    log("=" * 78)
    print_table(["id", "verdict", "time", "est.", "result"],
                [[r["id"], r["verdict"], C.fmt_t(r.get("elapsed")).strip(),
                  C.fmt_t(r.get("estimate")).strip(), r["headline"]] for r in results], out, "  ")
    rep = os.path.join(C.RESULTS, "report.html")
    write_report(rep, header, results)
    log("\nResults: %s/  (one JSON per experiment, PNG figures)" % os.path.relpath(C.RESULTS))
    log("Report:  %s  (open in a browser: summary, tables and every figure)" % os.path.relpath(rep))
    if nb:
        try:
            from IPython.display import display, HTML
            rows = "".join("<tr><td>%s</td><td><b style='color:%s'>%s</b></td><td>%s</td></tr>"
                           % (r["id"], VERDICT_COLOR.get(r["verdict"], "#898781"), r["verdict"],
                              html.escape(r["headline"])) for r in results)
            display(HTML("<table>%s</table>" % rows))
        except Exception:
            pass
    bad = [r for r in results if r["verdict"] == "FAIL"]
    return 1 if bad else 0


def report_one(e, recs, ctx, args, nb, log, elapsed, est):
    res = {"id": e.ID, "title": e.TITLE, "doc": e.__doc__, "elapsed": elapsed,
           "estimate": est, "figs": [],
           "errors": ["%s: %s" % (r.get("label", "?"), r["error"]) for r in recs if "error" in r]}
    try:
        res.update(e.analyze(recs, ctx))
    except Exception:
        res.update(verdict="FAIL", headline="analysis failed", table=None)
        res["errors"].append(traceback.format_exc())
    log("   -> %s  %s" % (res["verdict"], res["headline"]))
    if res.get("table"):
        print_table(res["table"]["cols"], res["table"]["rows"])
    for err in res["errors"]:
        log("   ERROR " + err.splitlines()[0][:300])
    if not args.no_plots and recs:
        try:
            plt = C.plt()
            figs = e.plot(recs, res, ctx, plt)
            for k, f in enumerate(figs):
                p = C.save_fig(f, "%s%s.png" % (e.ID, "" if k == 0 else "_%d" % k))
                res["figs"].append(fig_png(f))
                log("   figure: %s" % os.path.relpath(p))
                if nb:
                    plt.show()
                else:
                    plt.close(f)
        except ImportError:
            log("   (matplotlib not installed: no figures)")
        except Exception:
            res["errors"].append("plot failed:\n" + traceback.format_exc())
            log("   plot failed: " + traceback.format_exc().splitlines()[-1])
    log("")
    return res


if __name__ == "__main__":
    rc = main()
    if not C.in_notebook():
        sys.exit(rc)
