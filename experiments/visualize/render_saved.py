#!/usr/bin/env python3
"""Draw the devices of saved experiment results, without running anything.

    python3 experiments/visualize/render_saved.py experiments/results/R1.json [more.json ...]
        [--out experiments/results/devices]

Reads the records an experiment saved; every record that carries a
"geometry" (the run was made with --visualize true) gives one picture per
device.  Writes PNGs and a devices.html that shows them all.
"""

import argparse
import base64
import html
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def collect(records):
    """One geometry per device/scenario: the CPU run's when both backends recorded it."""
    out = {}
    for r in records:
        g = r.get("geometry")
        if not g or not g.get("devices"):
            continue
        name = r.get("scenario") or r.get("device") or r.get("label", "device").rsplit(" ", 1)[0]
        cpu = not (r.get("gpu_req") or r.get("gpu_requested"))
        if name not in out or cpu:
            out[name] = g
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "devices"))
    args = ap.parse_args(argv)
    import common as C
    from visualize import device
    plt = C.plt()
    os.makedirs(args.out, exist_ok=True)
    parts = []
    for path in args.results:
        with open(path) as f:
            d = json.load(f)
        for name, g in collect(d.get("records", [])).items():
            for k, fig in enumerate(device.render(g, "%s %s" % (d.get("id", ""), name), plt)):
                fn = os.path.join(args.out, "%s_%s%s.png" % (d.get("id", "x"), name.replace(" ", "_").replace("/", "_"),
                                                              "" if k == 0 else "_%d" % k))
                fig.savefig(fn, dpi=130, bbox_inches="tight")
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
                parts.append("<h2>%s</h2><img src='data:image/png;base64,%s'>"
                             % (html.escape("%s %s" % (d.get("id", ""), name)), base64.b64encode(buf.getvalue()).decode()))
                plt.close(fig)
    page = os.path.join(args.out, "devices.html")
    with open(page, "w") as f:
        f.write("<!doctype html><meta charset='utf-8'><title>Simulated devices</title>"
                "<style>body{font:14px system-ui;max-width:1200px;margin:auto;padding:16px}"
                "img{max-width:100%}</style><h1>Simulated devices</h1>" + "".join(parts))
    print("%d pictures; %s" % (len(parts), os.path.relpath(page)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
