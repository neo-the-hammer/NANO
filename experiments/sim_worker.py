#!/usr/bin/env python3
"""Run one scenario of sims.py under the backend the environment selects.

    python3 sim_worker.py NAME QUICK(0|1) OUT_JSON

Launched by common.run_sim() with VIDES_GPU / VIDES_PROFILE set, in a
scratch directory (the self-consistent solvers write ncar.ini / Phi.ini
into the working directory).  Counts and times every NEGF solve by
wrapping the solver entry points of the device classes.
"""

import faulthandler
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sims  # noqa: E402


def instrument(V):
    V._sim_solves = 0
    V._sim_t = 0.0
    for cname, meth in (("nanotube", "charge_T"), ("nanotube", "mode_charge_T"),
                        ("nanoribbon", "charge_T"), ("Hamiltonian", "charge_T"),
                        ("Zincblend", "charge_T")):
        cls = getattr(V, cname, None)
        f = getattr(cls, meth, None) if cls else None
        if f is None:
            continue

        def wrapped(self, _f=f):
            t0 = time.perf_counter()
            try:
                return _f(self)
            finally:
                V._sim_t += time.perf_counter() - t0
                V._sim_solves += 1
        setattr(cls, meth, wrapped)


def main():
    faulthandler.enable()      # a crash inside the C/Fortran code names the Python line
    name, quick, out = sys.argv[1], sys.argv[2] == "1", sys.argv[3]
    V = sims.load()
    instrument(V)
    t0 = time.perf_counter()
    try:
        res = sims.SCENARIOS[name](quick)
    except sims.Skip as e:
        res = {"skipped": str(e)}
    except Exception:
        res = {"error": traceback.format_exc()[-3000:]}
    res["wall"] = time.perf_counter() - t0
    res["solves"] = V._sim_solves
    res["t_solve"] = V._sim_t
    res["name"] = name
    with open(out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
