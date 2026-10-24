"""R2 -- the regression tests in test/, against their stored results.

The scripts behind test/test.py, each checked as the original does --
against the reference data shipped in test/ with the original tolerance
-- on the CPU and on the GPU, plus CPU vs GPU agreement.  This is the
"right answer" check: both backends must reproduce results computed long
before the GPU code existed.

  reg_gnr_fast_ohmic  test/script_GNR_auto.py        GNR path vs H path, same ribbon
  reg_graphene        test/test_demetrio_graphene_auto.py   vs g0.dat, g.dat
  reg_bilayer         test/test_bilayer_class_auto.py       vs carica.* (4 tests)
  reg_snw             test/test_SNW.py               sp3d5s* Si wires vs
                                                     transmission-test-Si*.dat
  reg_zincblend       test/script_test_sp3d5ss_auto.py      Si, Ge, InAs (skipped if
                                                     the Zincblend class does not
                                                     run under Python 3)
  reg_hole            test/test_hole.py              a vacancy built two ways

Tolerances CPU vs GPU: 1e-8 where both use the eigen self-energy (eta <
1e-6), 1e-7 with GPU decimation, 1e-5 where the methods differ (bilayer,
eta 1e-5).
"""

import _simexp as X

__doc__ += X.__doc__

ID = "R2"
TITLE = "Original regression tests (test/) vs stored references"
KIND = "correctness"
SCEN = [("reg_gnr_fast_ohmic", 1e-7, (3, 3)), ("reg_graphene", 1e-8, (30, 30)),
        ("reg_bilayer", 1e-5, (30, 30)), ("reg_snw", 1e-7, (10, 200)),
        ("reg_zincblend", 1e-7, (10, 30)), ("reg_hole", 1e-6, (3, 3))]


def steps(ctx):
    return X.make_steps(SCEN, ctx)


def analyze(recs, ctx):
    return X.analyze(recs, ctx, SCEN, "regression tests")


def plot(recs, summ, ctx, plt):
    return X.plot(recs, summ, ctx, plt, SCEN, "R2 regression")
