#!/usr/bin/env python3
"""Worker of E5: an ensemble of disordered graphene nanoribbons, one backend.

    python3 disorder_worker.py PAYLOAD_JSON OUT_JSON

PAYLOAD: {"n": 8, "lengths": [5, 10], "W": [0.5, 1.0], "samples": 8,
          "first_sample": 0, "grid": [lo, hi, dE], "eta": 1e-6}

Anderson disorder: every atom gets an on-site energy drawn uniformly from
[-W/2, W/2] (the ribbon's Phi), from a generator seeded by (L, W, sample),
so the CPU and the GPU see exactly the same ribbons.  W = 0 is the clean
ribbon (one sample).  Writes T(E) of every sample and the time of every
solve.
"""

import json
import sys
import time

import numpy as np


def seed(L, W, k):
    return int(1000003 * round(L * 10) + 7919 * round(W * 1000) + k) % (2 ** 31)


def main():
    pay = json.loads(sys.argv[1])
    out = sys.argv[2]
    from NanoTCAD_ViDES import nanoribbon
    lo, hi, dE = pay["grid"]
    res, times = [], []
    E = None
    for L in pay["lengths"]:
        for W in pay["W"]:
            ks = [0] if W == 0 else range(pay.get("first_sample", 0), pay.get("first_sample", 0) + pay["samples"])
            for k in ks:
                d = nanoribbon(int(pay["n"]), float(L))
                rng = np.random.default_rng(seed(L, W, k))
                d.Phi = W * (rng.random(np.size(d.z)) - 0.5)
                d.Elower, d.Eupper, d.dE = lo, hi, dE
                d.eta = pay.get("eta", 1e-6)
                d.mu1 = d.mu2 = 0.0
                t0 = time.perf_counter()
                d.charge_T()
                times.append(time.perf_counter() - t0)
                ne = int(round((hi - lo) / dE)) + 1
                T = np.asarray(d.T, float).ravel()[:ne]
                E = np.asarray(d.E, float).ravel()[:ne]
                res.append({"L": L, "W": W, "k": k, "T": T.tolist()})
    with open(out, "w") as f:
        json.dump({"E": E.tolist() if E is not None else [], "samples": res, "times": times,
                   "atoms": int(np.size(d.z)) if res else 0}, f)


if __name__ == "__main__":
    main()
