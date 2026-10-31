#!/usr/bin/env python3
"""Turn saved results into publication material: PDF figures and LaTeX tables.

Reads the JSON every experiment saved (experiments/results/*.json for the
CPU-vs-GPU suite, experiments/results/paper/*.json for the publication
suite), re-runs each experiment's own analysis, and writes to
experiments/results/paper/export/:

  <ID>.pdf, <ID>_1.pdf ...    the experiment's figures, vector, serif fonts
  <ID>.tex                    its tables as booktabs LaTeX
  summary.tex                 one row per experiment: verdict and result
  environment.tex             the machine and software manifest

Nothing is recomputed on the device; only the analysis and the drawing.

    python3 experiments/paper/export.py [--only C3 S7 E5] [--png]
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _runner  # noqa: E402,F401
import common as C  # noqa: E402

OUT = os.path.join(_runner.RESULTS, "export")
MAIN = os.path.join(os.path.dirname(HERE), "results")


def tex(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}"), ("+-", r"$\pm$"), ("—", "---")):
        s = s.replace(a, b)
    return s


def table_tex(t, caption, label):
    cols = t["cols"]
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{%s}" % tex(caption), r"\label{%s}" % label,
             r"\begin{tabular}{%s}" % ("l" * len(cols)), r"\toprule",
             " & ".join(r"\textbf{%s}" % tex(c) for c in cols) + r" \\", r"\midrule"]
    for row in t["rows"]:
        cells = [tex(c) for c in row] + [""] * (len(cols) - len(row))
        lines.append(" & ".join(cells[:len(cols)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def paper_style(plt):
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9,
                         "axes.labelsize": 9, "legend.fontsize": 8, "savefig.dpi": 300,
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def find(eid):
    for d in (MAIN, _runner.RESULTS):
        p = os.path.join(d, "%s.json" % eid)
        if os.path.exists(p):
            return p
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="+", metavar="ID")
    ap.add_argument("--png", action="store_true", help="also write 300-dpi PNGs")
    args = ap.parse_args(argv)
    import run_all
    import run_paper
    exps = run_all.EXPERIMENTS + run_paper.EXPERIMENTS
    if args.only:
        want = {x.upper() for x in args.only}
        exps = [e for e in exps if e.ID in want]
    os.makedirs(OUT, exist_ok=True)
    try:
        plt = C.plt()
        paper_style(plt)
    except ImportError:
        plt = None
    summary = []
    for e in exps:
        p = find(e.ID)
        if not p:
            continue
        with open(p) as f:
            d = json.load(f)
        ctx = {"quick": d.get("quick", False)}
        try:
            s = e.analyze(d["records"], ctx)
        except Exception as ex:
            print("%s: analysis failed: %s" % (e.ID, ex))
            continue
        summary.append((e.ID, e.TITLE, s["verdict"], s["headline"]))
        parts = []
        for k, t in enumerate([s.get("table")] + list(s.get("tables") or [])):
            if t and t.get("rows"):
                cap = "%s: %s" % (e.TITLE, t.get("title", "")) if t.get("title") else e.TITLE
                parts.append(table_tex(t, cap, "tab:%s%s" % (e.ID.lower(), "" if k == 0 else "_%d" % k)))
        with open(os.path.join(OUT, "%s.tex" % e.ID), "w") as f:
            f.write("\n".join(parts))
        if plt:
            try:
                figs = e.plot(d["records"], s, ctx, plt)
            except Exception as ex:
                print("%s: plot failed: %s" % (e.ID, ex))
                figs = []
            for k, fig in enumerate(figs):
                base = os.path.join(OUT, "%s%s" % (e.ID, "" if k == 0 else "_%d" % k))
                fig.savefig(base + ".pdf", bbox_inches="tight")
                if args.png:
                    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
                plt.close(fig)
        print("%-3s %-7s %s" % (e.ID, s["verdict"], s["headline"][:110]))
    with open(os.path.join(OUT, "summary.tex"), "w") as f:
        f.write(table_tex({"cols": ["ID", "experiment", "verdict", "result"],
                           "rows": [list(x) for x in summary]}, "Summary of the experiments", "tab:summary"))
    for d in (_runner.RESULTS, MAIN):
        env = os.path.join(d, "environment.json")
        if os.path.exists(env):
            sys.path.insert(0, HERE)
            import environment
            with open(env) as f:
                lines = environment.header_lines(json.load(f))
            with open(os.path.join(OUT, "environment.tex"), "w") as f:
                f.write("\\begin{itemize}\n" + "".join("  \\item %s\n" % tex(x) for x in lines) + "\\end{itemize}\n")
            break
    print("wrote %s" % os.path.relpath(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
