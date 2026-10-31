#!/usr/bin/env python3
"""Run the publication experiments with one command; writes report.html.

    python3 experiments/paper/run_paper.py                 # all, full sizes
    python3 experiments/paper/run_paper.py --quick         # small sizes
    python3 experiments/paper/run_paper.py --only E1 E5
    python3 experiments/paper/run_paper.py --replot        # re-analyse saved results
    python3 experiments/paper/run_paper.py --visualize true

In Colab, Kaggle or Jupyter:  %run experiments/paper/run_paper.py --quick

  E1  energy to solution (J per solve, GPU board power + CPU RAPL)
  E2  chunked energy batches (forced multi-chunk runs): same answers
  E3  scaling to large devices: largest ribbon per backend in a time budget
  E4  convergence in dE and eta; time to a 1e-3 accurate current
  E5  science case: Anderson localization over ensembles of disordered ribbons
  E6  multi-GPU: ky points of a 2D device split across GPUs
  E7  contact self-energy policy on the Hamiltonian path
  E8  timing statistics: repeated runs, 95% confidence intervals

Everything goes to experiments/results/paper/: one JSON per experiment,
PNG figures, environment.json (paper/environment.py) and report.html.
After a run, paper/export.py turns the saved results of both suites into
publication figures (PDF) and LaTeX tables.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _runner  # noqa: E402
import e1_energy, e2_chunking, e3_size_limit, e4_convergence  # noqa: E402,E401
import e5_disorder, e6_multi_gpu, e7_selfenergy_policy, e8_timing_stats  # noqa: E402,E401

EXPERIMENTS = [e1_energy, e2_chunking, e3_size_limit, e4_convergence, e5_disorder, e6_multi_gpu,
               e7_selfenergy_policy, e8_timing_stats]


def main(argv=None):
    return _runner.run(EXPERIMENTS, argv)


if __name__ == "__main__":
    rc = main()
    import common
    if not common.in_notebook():
        sys.exit(rc)
