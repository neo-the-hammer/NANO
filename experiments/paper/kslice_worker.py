#!/usr/bin/env python3
"""Worker of E6: part of the transverse-wave-vector sum of a graphene flake.

    python3 kslice_worker.py PAYLOAD_JSON OUT_JSON

PAYLOAD: {"L": 20.0, "k": [k1, k2, ...], "dk": dk, "grid": [lo, hi, dE],
          "phi": -0.1, "mu2": -0.2}

graphene.charge_T sums T(E, k) and the charge over k = kmin, kmin+dk, ...,
kmax with weight 2 dk / 2 pi.  Here each listed k is solved on its own
(kmin = kmax = k, same dk, so the same weight), and the partial sums are
returned; the parts of all workers add up to the full sum.  Which GPU
the worker uses is set by CUDA_VISIBLE_DEVICES.
"""

import json
import sys
import time

import numpy as np


def main():
    pay = json.loads(sys.argv[1])
    out = sys.argv[2]
    from NanoTCAD_ViDES import graphene
    lo, hi, dE = pay["grid"]
    ne = int(round((hi - lo) / dE)) + 1
    T = None
    charge = None
    t0 = time.perf_counter()
    for k in pay["k"]:
        d = graphene(float(pay["L"]))
        d.kmin = d.kmax = float(k)
        d.dk = float(pay["dk"])
        d.Elower, d.Eupper, d.dE = lo, hi, dE
        d.Phi = pay.get("phi", -0.1) * np.ones(np.size(d.Phi))
        d.mu2 = pay.get("mu2", -0.2)
        d.charge_T()
        t = np.asarray(d.T, float).ravel()[:ne]
        c = np.asarray(d.charge, float).ravel()
        T = t if T is None else T + t
        charge = c if charge is None else charge + c
    wall = time.perf_counter() - t0
    with open(out, "w") as f:
        json.dump({"T": T.tolist() if T is not None else [], "charge": charge.tolist() if charge is not None else [],
                   "nk": len(pay["k"]), "wall": wall}, f)


if __name__ == "__main__":
    main()
