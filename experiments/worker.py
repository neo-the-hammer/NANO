#!/usr/bin/env python3
"""Run one device end to end under the backend the environment selects.

    python3 worker.py SPEC_JSON REPEAT OUT_JSON

Launched by common.run_device() with VIDES_GPU / VIDES_SELFH /
VIDES_PROFILE set; the backend is chosen once per process, which is why
every (device, backend) pair gets its own process.

SPEC: {"kind": KIND, "size": [a, b], ...}

  KIND          class (entry point)                     size
  cnt           nanotube (CNT_charge_T)                 chirality n, length nm
  cntmode       nanotube, mode space (CNTmode_charge_T) chirality n, length nm
  cnt_schottky  nanotube with Schottky contacts         chirality n, length nm
  gnr           nanoribbon (GNR_charge_T)               width n, length nm
  gnr_schottky  nanoribbon with Schottky contacts       width n, length nm
  gnr_fast      nanoribbon_fast_ohmic (H_charge_T)      width n, length nm
  hamiltonian   Hamiltonian from demo/GNR.py            width, slices
  vacancy       Hamiltonian from test/ribbon5_hole.xyz  ignored
  snw           sp3d5s* Si nanowire (H_charge_T)        diameter A, length A
  graphene      graphene flake, summed over ky          length nm, number of ky
  bilayer       bilayer graphene, summed over ky        length nm, number of ky
  tmd           MoS2-like TMD flake (module_TMD)        length nm, number of ky

and
       "grid": [Elower, Eupper, dE]  (None: the class's own energy range),
       "phi": -0.2, "mu2": -0.3, "eta": 1e-5, "nmodes": 4}
"eta" applies to every kind when given (experiment E4 sweeps it).
"""

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Transverse-k structures: kF of graphene, and the default k windows of
# demo/tutorial_idvds_graphene.py and test/test_bilayer_class_auto.py.
ACC = 0.144
KF = 2 * np.pi / (3 * np.sqrt(3) * ACC)


def kwindow(d, kmin, kmax, nk):
    d.kmin, d.kmax = kmin, kmax
    d.dk = (kmax - kmin) / float(nk)


def build(spec):
    """Device classes of the library, one per kind (see the module doc)."""
    kind = spec["kind"]
    a, b = spec["size"]
    phi = spec.get("phi", -0.2)
    if kind in ("cnt", "cntmode", "cnt_schottky"):
        from NanoTCAD_ViDES import nanotube
        d = nanotube(int(a), float(b))
        if kind == "cntmode":
            d.Nmodes = int(spec.get("nmodes", 4))
        if kind == "cnt_schottky":
            d.contact = "Schottky"
        d.Phi = phi * np.ones(d.n * d.Nc)
    elif kind in ("gnr", "gnr_schottky"):
        from NanoTCAD_ViDES import nanoribbon
        d = nanoribbon(int(a), float(b))
        if kind == "gnr_schottky":
            d.contact = "Schottky"
        d.Phi = phi * d.z
    elif kind == "gnr_fast":
        from NanoTCAD_ViDES import nanoribbon_fast_ohmic
        d = nanoribbon_fast_ohmic(int(a), float(b))
        d.Phi = phi * d.z
    elif kind == "hamiltonian":
        from NanoTCAD_ViDES import Hamiltonian
        from GNR import GNR
        d = Hamiltonian(int(a), int(b))
        d.H = GNR(int(a), int(b))
        d.eta = spec.get("eta", 1e-5)
        d.Phi = phi * np.ones(d.n * d.Nc)
    elif kind == "vacancy":
        # test/ribbon5_hole.xyz: 5-wide ribbon with one atom removed
        from NanoTCAD_ViDES import Hamiltonian, get_xyz_from_file, create_H_from_xyz
        x, y, z = get_xyz_from_file(os.path.join(ROOT, "test", "ribbon5_hole.xyz"))
        r = create_H_from_xyz(x, y, z, 1, 0.0, -2.7, 1.45, 3)
        d = Hamiltonian(r[1], r[2])
        d.H = r[0]
        d.eta = spec.get("eta", 1e-5)
        d.Phi = phi * np.ones(d.n * d.Nc)
    elif kind == "snw":
        # sp3d5s* silicon nanowire (10 orbitals per atom, passivated),
        # as test/test_SNW.py: size = (diameter, length) in Angstrom
        from NanoTCAD_ViDES import (Hamiltonian, atoms_coordinates_nanowire, create_H_from_xyz,
                                    onsite_Si, thop_Si)
        x, y, z = atoms_coordinates_nanowire(np.array([5.431, 0, 1, float(a), float(b)]))
        r = create_H_from_xyz(np.asarray(x), np.asarray(y), np.asarray(z), 10, onsite_Si, thop_Si, 3, 4)
        d = Hamiltonian(r[1], r[2])
        d.H = r[0]
        d.eta = spec.get("eta", 1e-7)
        d.Phi = phi * np.ones(d.n * d.Nc)
    elif kind == "graphene":
        # 2D graphene flake, NEGF summed over ky: size = (L nm, number of ky)
        from NanoTCAD_ViDES import graphene
        d = graphene(float(a))
        kwindow(d, KF - 1.0, KF + 1.0, int(b))
        d.Phi = phi * np.ones(np.size(d.Phi))
    elif kind == "bilayer":
        from NanoTCAD_ViDES import bilayer_graphene
        d = bilayer_graphene(float(a))
        kwindow(d, 5.39722, 12.5958, int(b))
        d.eta = spec.get("eta", 1e-5)
        d.Phi = phi * np.ones(np.size(d.Phi))
    elif kind == "tmd":
        # MoS2-like transition-metal dichalcogenide flake (module_TMD.py),
        # summed over ky: size = (L nm, number of ky)
        from module_TMD import TMD
        d = TMD(float(a), "n")
        kwindow(d, 0.0, np.pi / d.delta, int(b))
        d.Phi = phi * np.ones(np.size(d.Phi))
    else:
        raise ValueError("unknown device kind %r" % kind)
    if spec.get("grid"):
        d.Elower, d.Eupper, d.dE = spec["grid"]
    d.mu2 = spec.get("mu2", -0.3)
    if "eta" in spec and kind not in ("hamiltonian", "vacancy", "snw", "bilayer"):
        d.eta = spec["eta"]           # (those four take it above, with their own defaults)
    return d


def solve(d, kind):
    if kind == "cntmode":
        d.mode_charge_T()
    else:
        d.charge_T()


def n_energies(d, spec):
    if spec.get("grid"):
        lo, hi, dE = spec["grid"]
        return int(round((hi - lo) / dE)) + 1
    E = np.asarray(d.E, float).ravel()
    k = 1
    while k < len(E) and E[k] > E[k - 1]:
        k += 1
    return k


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
    # The entry points fill the start of a preallocated array.
    ne = n_energies(d, spec)
    E = np.asarray(d.E, float).ravel()[:ne]
    T = np.asarray(d.T, float).ravel()[:ne]
    charge = np.asarray(d.charge, float).ravel()
    res = {"spec": spec, "times": times, "n": int(getattr(d, "n", 1)), "Nc": int(d.Nc),
           "NE": ne, "E": E.tolist(), "T": T.tolist(),
           "charge": charge.tolist(), "current": current}
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import visualize
    if visualize.enabled():                       # --visualize true: draw the device in the report
        from visualize import capture
        g = capture.device_geometry(d, spec["kind"])
        res["geometry"] = {"devices": [g] if g else [], "gates": [], "regions": [], "doping": []}
    with open(out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
