"""R1 -- the tutorials in demo/, run as real simulations on CPU and GPU.

Each tutorial script runs end to end, unchanged except as noted in
experiments/sims.py (Python 3, no plots, smaller in --quick), once per
backend.  This covers what a user actually runs: flat-band and biased
devices, a Laplace solution fed to the NEGF, and full Poisson-NEGF
self-consistent loops whose answer depends on every solve along the way.

  tut_cnt               demo/tutorial_CNT.py                CNT, defaults
  tut_gnr               demo/tutorial_GNR.py                GNR in a potential
  tut_hamilton          demo/tutorial_Hamilton.py           generic H (eigen on
                                                            CPU, decimation on GPU)
  tut_poisson_cnt       demo/tutorial_PoissonCNT.py         Laplace + NEGF, CNT FET
  tut_scf_cnt           demo/tutorial_selfconsistent.py     SCF, CNT mode space
  tut_scf_gnr_schottky  demo/tutorial_selfconsistentGNRschottky.py
                                                            SCF, GNR, Schottky
  tut_idvgs             demo/tutorial_idvgs.py              Id(Vgs) sweep, SCF per bias
  tut_graphene_idvds    demo/tutorial_idvds_graphene.py     Id(Vds), 2D graphene FET,
                                                            NEGF per ky (reduced)

Tolerances: 1e-8 same arithmetic; 1e-7 with GPU decimation; 1e-5 where the
backends use different self-energy methods (tut_hamilton); 1e-6 for the
self-consistent loops, which stop at a convergence threshold (if the two
backends need a different number of iterations this is reported).
Laws: 0 <= T <= channels, finite results; for Id(Vgs) a current rising
with the gate and a subthreshold swing no better than 60 mV/dec.
"""

import _simexp as X

__doc__ += X.__doc__

ID = "R1"
TITLE = "Original tutorials (demo/) as real simulations: CPU vs GPU"
KIND = "correctness"
SCEN = [("tut_cnt", 1e-8, (2, 2)), ("tut_gnr", 1e-7, (1, 1)), ("tut_hamilton", 1e-5, (2, 2)),
        ("tut_poisson_cnt", 1e-8, (10, 10)), ("tut_scf_cnt", 1e-6, (30, 120)),
        ("tut_scf_gnr_schottky", 1e-6, (20, 60)), ("tut_idvgs", 1e-6, (60, 120)),
        ("tut_graphene_idvds", 1e-6, (60, 400))]


def steps(ctx):
    return X.make_steps(SCEN, ctx)


def analyze(recs, ctx):
    return X.analyze(recs, ctx, SCEN, "tutorials")


def plot(recs, summ, ctx, plt):
    return X.plot(recs, summ, ctx, plt, SCEN, "R1 tutorials")
