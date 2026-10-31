"""Shared entry point of the publication experiments.

Each experiment in this folder is a module with the interface of the
CPU-vs-GPU suite (ID, TITLE, KIND, steps, analyze, plot; see
experiments/exps/c1_rgf.py) and runs through the same machinery as
experiments/run_all.py: calibration, plan with ETA, progress, verdicts,
tables, figures, and a report.html when it finishes.  Results go to
experiments/results/paper/ so they never overwrite the main suite's.

    python3 experiments/paper/run_paper.py [--quick] [--only E1 E5] [--visualize true]
    python3 experiments/paper/e1_energy.py [--quick]        # one experiment alone
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
for p in (HERE, EXP, os.path.join(EXP, "exps")):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS = os.path.join(EXP, "results", "paper")
TITLE = "NanoTCAD ViDES — publication experiments"


def run(experiments, argv=None, needs_module=None):
    import run_all
    if needs_module is None:
        needs_module = {e.ID for e in experiments if getattr(e, "NEEDS_MODULE", True)}
    return run_all.main(argv, experiments=experiments, needs_module=needs_module,
                        results_dir=RESULTS, title=TITLE,
                        description="Publication experiments of the NanoTCAD ViDES GPU backend")


def run_one(module, argv=None):
    rc = run([module], argv)
    import common
    if not common.in_notebook():
        sys.exit(rc)
    return rc
