#!/usr/bin/env python3
"""Plot the results of test/benchmark_gpu.py.

    python3 test/plot_benchmark.py benchmark_results.json
    python3 test/plot_benchmark.py benchmark_results.json --out bench.png

In a Colab / Jupyter cell, to show the figure inline as well as saving it:

    %run ../test/plot_benchmark.py benchmark_results.json --show

One column per device path (CNT, GNR, Hamiltonian).  Devices are ordered
left to right by the amount of work in one solve, NE * Nc * n^3 (energy
points x blocks x cost of one n x n inversion).

  top row     time per solve, log scale: CPU, GPU steady state, and the
              first GPU call (which includes one-off CUDA start-up).
  bottom row  speedup = CPU / GPU steady.  Above the 1x line the GPU wins.

The two rows are separate charts on purpose: times and speedups are
different quantities and do not share an axis.
"""

import argparse
import json
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Reference palette, validated for colour-vision deficiency on this
# surface: two categorical slots (blue, orange) plus neutral chrome.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CPU_C = "#2a78d6"    # slot 1
GPU_C = "#eb6834"    # slot 2

PATH_TITLES = {
    "cnt": "Carbon nanotube",
    "gnr": "Graphene nanoribbon",
    "hamiltonian": "Hamiltonian / nanowire (Lake)",
}
PATH_ORDER = ["cnt", "gnr", "hamiltonian"]


def fmt_time(t, _pos=None):
    if t >= 60:
        return "%g min" % float("%.2g" % (t / 60))
    if t >= 1:
        return "%g s" % float("%.2g" % t)
    if t >= 1e-3:
        return "%g ms" % float("%.2g" % (t * 1e3))
    return "%g µs" % float("%.2g" % (t * 1e6))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def size_label(r):
    return "n=%d\nNc=%d" % (r["n"], r["Nc"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", nargs="?", default="benchmark_results.json")
    ap.add_argument("--out", default="benchmark.png",
                    help="output image; .png, .svg or .pdf (default benchmark.png)")
    ap.add_argument("--show", action="store_true",
                    help="also display the figure (Jupyter / Colab)")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    with open(args.results) as f:
        data = json.load(f)
    rows = data.get("rows", [])
    if not rows:
        sys.exit("no results in %s" % args.results)

    paths = [p for p in PATH_ORDER if any(r["path"] == p for r in rows)]
    paths += sorted({r["path"] for r in rows} - set(paths))

    ncol = len(paths)
    fig, axes = plt.subplots(
        2, ncol, figsize=(4.2 * ncol + 0.6, 7.2), squeeze=False,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.55, "wspace": 0.28})
    fig.patch.set_facecolor(SURFACE)

    # Shared time scale across columns so paths compare honestly.
    all_t = [r[k] for r in rows for k in ("cpu_s", "gpu_steady_s", "gpu_first_s")]
    tmin, tmax = min(all_t), max(all_t)
    all_sp = [r["speedup"] for r in rows]
    spmax = max(max(all_sp), 1.0) * 1.25

    for c, path in enumerate(paths):
        pr = sorted((r for r in rows if r["path"] == path),
                    key=lambda r: r["NE"] * r["Nc"] * r["n"] ** 3)
        x = list(range(len(pr)))
        labels = [size_label(r) for r in pr]

        # ---- time per solve -------------------------------------------
        ax = axes[0][c]
        style(ax)
        ax.set_yscale("log")
        ax.set_ylim(tmin / 2.5, tmax * 2.5)
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_time))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))

        cpu = [r["cpu_s"] for r in pr]
        gpu = [r["gpu_steady_s"] for r in pr]
        first = [r["gpu_first_s"] for r in pr]

        ax.plot(x, first, color=GPU_C, linewidth=1.5, linestyle=(0, (3, 2)),
                marker="o", markersize=7, markerfacecolor=SURFACE,
                markeredgewidth=1.5, label="GPU, first call", zorder=2)
        ax.plot(x, cpu, color=CPU_C, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label="CPU",
                zorder=3)
        ax.plot(x, gpu, color=GPU_C, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2,
                label="GPU, steady state", zorder=3)

        # Direct labels at the largest device only, in ink not series colour.
        for val, dy in ((cpu[-1], 1), (gpu[-1], -1)):
            ax.annotate(fmt_time(val), (x[-1], val), xytext=(8, 4 * dy),
                        textcoords="offset points", fontsize=8, color=INK_2,
                        va="center")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.4, len(pr) - 0.4 + 0.5)
        ax.set_title(PATH_TITLES.get(path, path), loc="left", fontsize=11,
                     color=INK, pad=10)
        if c == 0:
            ax.set_ylabel("time per solve", fontsize=9, color=INK_2)

        # ---- speedup ----------------------------------------------------
        ax = axes[1][c]
        style(ax)
        sp = [r["speedup"] for r in pr]
        bars = ax.bar(x, sp, width=0.55, color=GPU_C, edgecolor=SURFACE,
                      linewidth=2, zorder=2)
        ax.axhline(1.0, color=INK_2, linewidth=1, zorder=3)
        ax.annotate("1x  (no gain)", (len(pr) - 0.4 + 0.45, 1.0),
                    xytext=(0, 4), textcoords="offset points",
                    fontsize=7.5, color=MUTED, ha="right", va="bottom")
        for b, v in zip(bars, sp):
            ax.annotate("%.1fx" % v, (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_ylim(0, spmax)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: "%gx" % v))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.4, len(pr) - 0.4 + 0.5)
        if c == 0:
            ax.set_ylabel("speedup (CPU / GPU)", fontsize=9, color=INK_2)

    # One legend for the time row, above the charts.
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    order = [labels_.index(l) for l in ("CPU", "GPU, steady state", "GPU, first call")]
    fig.legend([handles[i] for i in order], [labels_[i] for i in order],
               loc="upper left", bbox_to_anchor=(0.06, 0.935), ncol=3,
               frameon=False, fontsize=9, labelcolor=INK_2,
               handlelength=2.6)

    backend = data.get("backend", "").replace("NEGF backend:", "").strip()
    fig.suptitle("NanoTCAD ViDES NEGF solve: CPU vs GPU",
                 x=0.06, y=0.995, ha="left", fontsize=13, color=INK)
    sub = "Devices ordered by work per solve (energy points x blocks x n^3)."
    if backend:
        sub = "GPU: %s.  %s" % (backend, sub)
    fig.text(0.06, 0.955, sub, fontsize=8.5, color=MUTED, ha="left")

    fig.savefig(args.out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("saved %s" % args.out)

    # The same numbers as a table, so nothing is read off a chart alone.
    print()
    print("%-12s %5s %5s %5s  %10s %10s %10s %8s"
          % ("path", "n", "Nc", "NE", "CPU", "GPU", "GPU 1st", "speedup"))
    for path in paths:
        for r in sorted((r for r in rows if r["path"] == path),
                        key=lambda r: r["NE"] * r["Nc"] * r["n"] ** 3):
            print("%-12s %5d %5d %5d  %10s %10s %10s %7.1fx"
                  % (path, r["n"], r["Nc"], r["NE"], fmt_time(r["cpu_s"]),
                     fmt_time(r["gpu_steady_s"]), fmt_time(r["gpu_first_s"]),
                     r["speedup"]))

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
