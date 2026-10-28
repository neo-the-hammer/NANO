#!/usr/bin/env python3
"""Find out why a Zincblend wire stalls or crashes on the GPU.

Runs the Si wire of reg_zincblend in a fresh process under several
settings (each switches one part of the concurrency layer off) with a
time limit, and prints which settings finish.  From the ViDES directory:

    python3 experiments/diag_zincblend.py [seconds per run, default 120]
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CODE = r"""
import faulthandler, os, sys, time
faulthandler.enable()
faulthandler.dump_traceback_later(float(os.environ["LIM"]) - 5, exit=True)
sys.path[:0] = [os.path.join(os.environ["ROOT"], "src")]
import numpy as np
import NanoTCAD_ViDES as V
t = time.time()
H = V.Zincblend("Si", 0, 1, 3.85, 0.0)
h = np.asarray(H.H)
print("built: n=%d Nc=%d orbitals=%d rows=%d  %.1fs" % (H.n, H.Nc, int(np.real(h[0, 0])), h.shape[0], time.time() - t), flush=True)
print("H checksum: |H|=%.10g  sum=%.10g  nan=%d" % (np.abs(h).sum(), np.real(h).sum(), int(np.isnan(h).sum())), flush=True)
H.Elower, H.Eupper, H.dE, H.eta = 0, 10, 0.1, 1e-8
H.mu1 = H.mu2 = 0.0
t = time.time()
H.charge_T()
T = np.asarray(H.T[:101], float)
ref = np.loadtxt(os.path.join(os.environ["ROOT"], "test", "transmission-test-Si.dat"))[:, 1]
m = min(len(ref), len(T))
print("solved in %.2fs, max T = %.4f, T checksum %.8g, max|T - stored| = %.3g, non-integer T at %d of %d energies"
      % (time.time() - t, T.max(), T.sum(), np.abs(T[:m] - ref[:m]).max(), int(np.sum(np.abs(T - np.round(T)) > 1e-3)), len(T)), flush=True)
"""

VARIANTS = [
    ("CPU only", {"VIDES_GPU": "0"}),
    ("CPU only, again", {"VIDES_GPU": "0"}),
    ("CPU only, 1 thread", {"VIDES_GPU": "0", "VIDES_THREADS": "1"}),
    ("GPU, defaults", {"VIDES_GPU": "1"}),
    ("GPU, pipeline off", {"VIDES_GPU": "1", "VIDES_PIPELINE": "0"}),
    ("GPU, hybrid off", {"VIDES_GPU": "1", "VIDES_HYBRID": "0"}),
    ("GPU, hybrid and pipeline off", {"VIDES_GPU": "1", "VIDES_HYBRID": "0", "VIDES_PIPELINE": "0"}),
    ("GPU, 1 CPU thread", {"VIDES_GPU": "1", "VIDES_THREADS": "1"}),
    ("GPU, memory pool off", {"VIDES_GPU": "1", "VIDES_GPU_POOL": "0"}),
    ("GPU, decimation self-energy", {"VIDES_GPU": "1", "VIDES_SELFH": "dec"}),
]


def main():
    lim = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    for name, extra in VARIANTS:
        env = dict(os.environ, ROOT=ROOT, LIM=str(lim), **extra)
        try:
            p = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True,
                               text=True, timeout=lim, cwd=os.path.join(ROOT, "src"))
            out = (p.stdout + p.stderr).strip().splitlines()
            keep = [l for l in out if l.startswith(("built", "H checksum", "solved", "File", "Thread", "Fatal", "Current"))
                    or "Segmentation" in l]
            status = "ok" if p.returncode == 0 else "rc=%d" % p.returncode
        except subprocess.TimeoutExpired:
            keep, status = [], "HUNG"
        print("%-32s %s" % (name, status))
        for l in keep[-8:]:
            print("    " + l[:160])
    return 0


if __name__ == "__main__":
    sys.exit(main())
