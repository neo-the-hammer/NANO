"""Device visualization for the experiment reports (optional).

Off by default.  Turn it on with

    python3 experiments/run_all.py --visualize true        (or false)
    python3 experiments/paper/run_paper.py --visualize true
    VIDES_VISUALIZE=true python3 ...                        (any runner)

When it is on, the worker processes record the geometry of every device
they solve (capture.py): atom positions, the potential and charge on each
atom, and, for self-consistent devices, the gates, dielectric regions and
doped reservoirs.  After an experiment's own figures, the report shows one
picture per device (device.py): a side view with the gates, oxide and
doping, a top view, and the potential and charge profiles along the
transport direction.  render_saved.py draws them again from saved results.
"""

import os


def enabled():
    v = os.environ.get("VIDES_VISUALIZE", "false").strip().lower()
    return v in ("1", "true", "yes", "on", "y", "t")
