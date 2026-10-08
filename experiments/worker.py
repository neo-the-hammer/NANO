#!/usr/bin/env python3
"""Run one device end to end under the backend the environment selects.

    python3 worker.py SPEC_JSON REPEAT OUT_JSON

Launched by common.run_device() with VIDES_GPU / VIDES_SELFH /
VIDES_PROFILE set; the backend is chosen once per process, which is why
every (device, backend) pair gets its own process.

SPEC: {"kind": "cnt" | "cntmode" | "gnr" | "hamiltonian",
       "size": [a, b],               (chirality, length nm) | (width, length nm)
                                     | (width, slices) for hamiltonian
       "grid": [Elower, Eupper, dE],
       "phi": -0.2, "mu2": -0.3, "eta": 1e-5, "nmodes": 4}
"""

import json
import os
import sys
import time

import numpy as np


def build(spec):
    kind = spec["kind"]
    a, b = spec["size"]
    phi = spec.get("phi", -0.2)
    if kind in ("cnt", "cntmode"):
        from NanoTCAD_ViDES import nanotube
        d = nanotube(int(a), float(b))
        if kind == "cntmode":
            d.Nmodes = int(spec.get("nmodes", 4))
        d.Phi = phi * np.ones(d.n * d.Nc)
    elif kind == "gnr":
        from NanoTCAD_ViDES import nanoribbon
        d = nanoribbon(int(a), float(b))
        d.Phi = phi * d.z
    elif kind == "hamiltonian":
        from NanoTCAD_ViDES import Hamiltonian
        from GNR import GNR
        d = Hamiltonian(int(a), int(b))
        d.H = GNR(int(a), int(b))
        d.eta = spec.get("eta", 1e-5)
        d.Phi = phi * np.ones(d.n * d.Nc)
    else:
        raise ValueError("unknown device kind %r" % kind)
    d.Elower, d.Eupper, d.dE = spec["grid"]
    d.mu2 = spec.get("mu2", -0.3)
    return d


def solve(d, kind):
    if kind == "cntmode":
        d.mode_charge_T()
    else:
        d.charge_T()


def main():
    spec = json.loads(sys.argv[1])
    repeat = max(1, int(sys.argv[2]))
    out = sys.argv[3]
    times = []
    d = None
    for _ in range(repeat):
        d = build(spec)          # fresh device each time: same work
        t0 = time.perf_counter()
        solve(d, spec["kind"])
        times.append(time.perf_counter() - t0)
    current = float(d.current())
    # H_charge_T fills the start of a preallocated array; count from the grid.
    ne = int(round((d.Eupper - d.Elower) / d.dE)) + 1
    E = np.asarray(d.E, float).ravel()[:ne]
    T = np.asarray(d.T, float).ravel()[:ne]
    charge = np.asarray(d.charge, float).ravel()
    with open(out, "w") as f:
        json.dump({"spec": spec, "times": times, "n": int(d.n), "Nc": int(d.Nc),
                   "NE": ne, "E": E.tolist(), "T": T.tolist(),
                   "charge": charge.tolist(), "current": current}, f)


if __name__ == "__main__":
    main()
