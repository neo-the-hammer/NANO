"""P1 -- device physics: laws every correct NEGF result obeys.

Designed checks whose right answer is known without any reference run,
evaluated on each backend, plus CPU vs GPU agreement of every curve.

  phys_ballistic          flat-band CNT, GNR and H-path ribbon: T(E) is an
                          integer (the number of open modes) and T(E) = T(-E)
                          (electron-hole symmetry)
  phys_shift_reciprocity  a uniform potential shifts T(E) rigidly; mirroring
                          an asymmetric potential leaves T(E) unchanged
  phys_neutral            no net charge at flat band, mu = 0; charge induced
                          by a potential
  phys_iv                 I(Vds = 0) = 0, |I| monotonic in Vds, I falling
                          as a barrier is raised (CNT and GNR)

Tolerances CPU vs GPU: 1e-7, and 1e-5 for phys_ballistic (its H-path
ribbon uses eigen on the CPU and decimation on the GPU).
"""

import _simexp as X

__doc__ += X.__doc__

ID = "P1"
TITLE = "Device physics laws on CPU and GPU"
KIND = "correctness"
SCEN = [("phys_ballistic", 1e-5, (5, 15)), ("phys_shift_reciprocity", 1e-7, (8, 20)),
        ("phys_neutral", 1e-7, (3, 3)), ("phys_iv", 1e-7, (15, 40))]


def steps(ctx):
    return X.make_steps(SCEN, ctx)


def analyze(recs, ctx):
    return X.analyze(recs, ctx, SCEN, "physics scenarios")


def plot(recs, summ, ctx, plt):
    return X.plot(recs, summ, ctx, plt, SCEN, "P1 physics")
