"""Real simulations for the CPU-vs-GPU comparison: the original scripts.

Every scenario here is one of the scripts that ship with ViDES -- the
tutorials in demo/ and the regression tests in test/ -- rewritten only as
far as needed to run unattended under Python 3 (no plots, no prints, no
MPI, results returned instead of written), plus a few designed physics
checks.  Each scenario says which script it comes from and what changed.

A scenario is a function scenario(quick) returning a dict:

  series   {name: [floats]}            compared CPU vs GPU (rel. to peak)
  x        {name: [floats]}            optional x axis of a series (plots)
  scalars  {name: float}               compared CPU vs GPU (relative)
  refs     {name: {err, tol, what}}    error against stored reference data
                                       shipped in test/ (the "right answer")
  laws     {name: {err, tol, what}}    physics the result must obey
  info     {name: value}               reported only

They are run by sim_worker.py, one (scenario, backend) per process, and
judged by exps/_simexp.py.  Needs NumPy and the built module.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEST = os.path.join(ROOT, "test")
DEMO = os.path.join(ROOT, "demo")

V = None    # the NanoTCAD_ViDES module, imported by load()


def load():
    global V
    if V is None:
        if DEMO not in sys.path:
            sys.path.append(DEMO)
        import NanoTCAD_ViDES
        V = NanoTCAD_ViDES
    return V


class Skip(Exception):
    """The scenario cannot run here, for a reason that is not the backend's."""


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def ne_of(E):
    """Number of energies actually computed: E, T are preallocated to
    NEmax and filled from the start with an increasing grid."""
    E = np.asarray(E, float).ravel()
    n = 1
    while n < len(E) and E[n] > E[n - 1]:
        n += 1
    return n


def lst(a):
    return [float(v) for v in np.asarray(a, float).ravel()]


def law(err, tol, what):
    return {"err": float(err), "tol": float(tol), "what": what}


# Per-atom charge is compared against max(peak, CHARGE_SCALE), a typical
# per-atom charge at a few tenths of a volt of gating: at flat band the net
# charge is a cancellation residue (~1e-7, phys_neutral), so its own peak
# is no yardstick.
CHARGE_SCALE = 1e-3


def dev_result(d, name="", nch=None, equilibrium=False):
    """E, T, charge and current of a solved device, plus the laws every
    coherent two-terminal result obeys: 0 <= T <= channels, finite.

    equilibrium=True (flat band, mu1 = mu2, nothing gated): the net
    per-atom charge is then the cancellation of electron and hole terms
    that are themselves dominated by states at E ~ 0 (zigzag edge states,
    an isolated vacancy level), where the LDOS ~ 1/eta is ill-conditioned:
    two correct solvers differ there by ~1e-7 absolute while T agrees to
    1e-14 (test_gpu_vs_cpu.py, case hamiltonian_eq).  That charge carries
    no physics -- neutrality itself is checked by phys_neutral -- so it is
    reported ("aux") but not judged.  Charge is judged in every biased or
    gated scenario."""
    p = name + ":" if name else ""
    k = ne_of(d.E)
    E, T = np.asarray(d.E, float)[:k], np.asarray(d.T, float)[:k]
    ch = np.asarray(d.charge, float).ravel()
    I = float(d.current())
    out = {"series": {p + "T": lst(T)},
           "aux": {p + "charge": lst(ch)} if equilibrium else {},
           "x": {p + "T": lst(E)},
           "scalars": {p + "current": I},
           "scale": {p + "charge": CHARGE_SCALE},
           "laws": {}}
    if not equilibrium:
        out["series"][p + "charge"] = lst(ch)
    peak = max(1.0, float(np.max(np.abs(T)))) if T.size else 1.0
    out["laws"][p + "T >= 0"] = law(max(0.0, -float(np.min(T))) / peak if T.size else 0, 1e-6,
                                    "transmission is never negative")
    if nch:
        out["laws"][p + "T <= channels"] = law(max(0.0, float(np.max(T)) - nch), 1e-6,
                                               "T never exceeds the %d channels of a slice" % nch)
    bad = int(np.sum(~np.isfinite(T)) + np.sum(~np.isfinite(ch)) + (0 if np.isfinite(I) else 1))
    out["laws"][p + "finite"] = law(bad, 0.5, "no NaN/inf in T, charge or current")
    return out


def merge(*parts):
    out = {}
    for p in parts:
        for k, v in p.items():
            if isinstance(v, dict):
                out.setdefault(k, {}).update(v)
            else:
                out[k] = v
    return out


def ref_maxabs(calc, ref, tol, what):
    calc = np.asarray(calc, float).ravel()
    ref = np.asarray(ref, float).ravel()
    m = min(len(calc), len(ref))
    return {"err": float(np.max(np.abs(calc[:m] - ref[:m]))), "tol": tol, "what": what}


def solves():
    return getattr(load(), "_sim_solves", 0)


# ======================================================================
# R1 -- the tutorials in demo/
# ======================================================================

def tut_cnt(quick):
    """demo/tutorial_CNT.py: a (13,0) CNT, 2 nm, every default (flat
    band, default energy range, eta 1e-5).  Unchanged."""
    V = load()
    CNT = V.nanotube(13, 2)
    CNT.charge_T()
    return dev_result(CNT, nch=CNT.n, equilibrium=True)


def tut_gnr(quick):
    """demo/tutorial_GNR.py: GNR n=3, 1 nm, Phi = -0.2 z, mu2 = -0.3,
    E in [-3, 3] step 0.05.  Unchanged."""
    V = load()
    G = V.nanoribbon(3, 1)
    G.Elower, G.Eupper, G.dE = -3, 3, 0.05
    G.Phi = -0.2 * G.z
    G.mu2 = -0.3
    G.charge_T()
    return dev_result(G, nch=G.n)


def tut_hamilton(quick):
    """demo/tutorial_Hamilton.py: a 5x8 ribbon built by demo/GNR.py
    through the generic Hamiltonian path, eta 1e-5, E in [-3, 3] step
    0.05.  Unchanged, so each backend uses its own default self-energy
    (eigen on the CPU, decimation on the GPU)."""
    V = load()
    from GNR import GNR
    H = V.Hamiltonian(5, 8)
    H.H = GNR(5, 8)
    H.Elower, H.Eupper, H.dE = -3, 3, 0.05
    H.eta = 1e-5
    H.charge_T()
    return dev_result(H, nch=H.n, equilibrium=True)


def tut_poisson_cnt(quick):
    """demo/tutorial_PoissonCNT.py: Laplace solution of a double-gate
    (13,0) CNT FET, 10 nm, then the NEGF in that potential.  Unchanged."""
    V = load()
    CNT = V.nanotube(13, 10)
    xg = V.linspace(-1, 1, 20)
    grid = V.grid3D(xg, xg, CNT.z, CNT.x, CNT.y, CNT.z)
    zlo, zhi = grid.gridz[0], grid.gridz[grid.nz - 1]
    top = V.gate("hex", 1, 1, -1, 1, zlo, zhi)
    bot = V.gate("hex", -1, -1, -1, 1, zlo, zhi)
    top.Ef, bot.Ef = -0.1, 0
    ox = V.region("hex", -1, 1, -1, 1, 0, zhi)
    ox.eps = 3.9
    p = V.interface3D(grid, top, bot, ox)
    p.normpoisson = 1e-1
    V.solve_Poisson(grid, p)
    CNT.Phi = p.Phi[grid.swap]
    CNT.charge_T()
    out = dev_result(CNT, nch=CNT.n)
    out["series"]["Phi on CNT"] = lst(CNT.Phi)
    return out


def _scf_out(d, grid, p, nch):
    out = dev_result(d, nch=nch)
    out["series"]["Phi on channel"] = lst(p.Phi[grid.swap])
    out["info"] = {"SCF iterations": solves()}
    return out


def tut_scf_cnt(quick):
    """demo/tutorial_selfconsistent.py: double-gate (13,0) CNT FET,
    Poisson-NEGF self-consistent loop in mode space (2 modes), top gate
    -0.6 V, mu2 = -0.5.  Full: 30 nm as written.  Quick: 10 nm (grid end
    moved with the tube)."""
    V = load()
    x1 = np.array([-2, -1.70172, -1.43483, -1.19604, -0.982388, -0.791223, -0.620181, 0])
    x2 = np.array([0.620181, 0.791223, 0.982388, 1.19604, 1.43483, 1.70172, 2])
    xg = np.concatenate((x1, x2))
    yg = np.concatenate((x1[:-1], x2))
    L = 10 if quick else 30
    CNT = V.nanotube(13, L)
    zg = np.array([-0.072, 30.024]) if not quick else np.array([-0.072, float(np.max(CNT.z)) + 0.072])
    grid = V.grid3D(xg, yg, zg, CNT.x, CNT.y, CNT.z)
    top = V.gate("hex", grid.xmax, grid.xmax, grid.ymin, grid.ymax, grid.zmin, grid.zmax)
    bot = V.gate("hex", grid.xmin, grid.xmin, grid.ymin, grid.ymax, grid.zmin, grid.zmax)
    top.Ef, bot.Ef = -0.6, 0
    CNT.Nmodes = 2
    CNT.mu2 = -0.5
    ox = V.region("hex", -2, 2, -2, 2, grid.zmin, grid.zmax)
    ox.eps = 3.9
    p = V.interface3D(grid, top, bot, ox)
    p.normpoisson = 1e-3
    V.solve_Poisson(grid, p)
    p.modespace = "yes"
    p.normpoisson = 1e-1
    p.normd = 5e-2
    V.solve_self_consistent(grid, p, CNT)
    return _scf_out(CNT, grid, p, 2 * CNT.Nmodes)


def tut_scf_cnt_tight(quick):
    """tut_scf_cnt (demo/tutorial_selfconsistent.py, 10 nm) converged
    tightly: Poisson and self-consistency thresholds 1e-6 instead of
    1e-1 / 5e-2.  The tutorial's loose thresholds let CPU and GPU stop at
    slightly different points; converged, both must land on the same
    solution to near round-off."""
    V = load()
    x1 = np.array([-2, -1.70172, -1.43483, -1.19604, -0.982388, -0.791223, -0.620181, 0])
    x2 = np.array([0.620181, 0.791223, 0.982388, 1.19604, 1.43483, 1.70172, 2])
    CNT = V.nanotube(13, 10)
    zg = np.array([-0.072, float(np.max(CNT.z)) + 0.072])
    grid = V.grid3D(np.concatenate((x1, x2)), np.concatenate((x1[:-1], x2)), zg, CNT.x, CNT.y, CNT.z)
    top = V.gate("hex", grid.xmax, grid.xmax, grid.ymin, grid.ymax, grid.zmin, grid.zmax)
    bot = V.gate("hex", grid.xmin, grid.xmin, grid.ymin, grid.ymax, grid.zmin, grid.zmax)
    top.Ef, bot.Ef = -0.6, 0
    CNT.Nmodes = 2
    CNT.mu2 = -0.5
    ox = V.region("hex", -2, 2, -2, 2, grid.zmin, grid.zmax)
    ox.eps = 3.9
    p = V.interface3D(grid, top, bot, ox)
    p.normpoisson = 1e-6
    V.solve_Poisson(grid, p)
    p.modespace = "yes"
    p.normd = 1e-6
    V.solve_self_consistent(grid, p, CNT)
    return _scf_out(CNT, grid, p, 2 * CNT.Nmodes)


def tut_scf_gnr_schottky(quick):
    """demo/tutorial_selfconsistentGNRschottky.py: double-gate GNR FET
    (n=6, 1.37 nm wide) with Schottky contacts, self-consistent, gates
    -0.3 V, mu2 = -0.2.  Full: 15 nm as written.  Quick: 6 nm."""
    V = load()
    G = V.nanoribbon(6, 6 if quick else 15)
    xg = V.nonuniformgrid(np.array([-2, 0.3, 0, 0.2, 2, 0.3]))
    yg = V.nonuniformgrid(np.array([-1, 0.3, 0, 0.2, 1.37, 0.2, 2.37, 0.3]))
    grid = V.grid3D(xg, yg, G.z, G.x, G.y, G.z)
    G.contact = "Schottky"
    top = V.gate("hex", grid.xmax, grid.xmax, grid.ymin, grid.ymax, grid.zmin, grid.zmax)
    bot = V.gate("hex", grid.xmin, grid.xmin, grid.ymin, grid.ymax, grid.zmin, grid.zmax)
    top.Ef = bot.Ef = -0.3
    G.mu2 = -0.2
    ox = V.region("hex", -2, 2, -2, 2, grid.zmin, grid.zmax)
    ox.eps = 3.9
    p = V.interface3D(grid, top, bot, ox)
    p.normpoisson = 1e-3
    V.solve_Poisson(grid, p)
    p.normpoisson = 1e-1
    p.normd = 5e-2
    V.solve_self_consistent(grid, p, G)
    return _scf_out(G, grid, p, G.n)


def tut_idvgs(quick):
    """demo/tutorial_idvgs.py: transfer characteristic Id(Vgs) of a
    doped-reservoir (13,0) CNT FET, 15 nm, mode space (2 modes),
    Vds = 0.1 V, self-consistent at every bias.  Full: Vgs 0..0.5 step
    0.05 as written.  Quick: step 0.1.  The Vgs list is explicit (the
    original accumulates a float step)."""
    V = load()
    CNT = V.nanotube(13, 15)
    x = V.nonuniformgrid(np.array([-2, 0.3, 0, 0.2, 2, 0.3]))
    grid = V.grid3D(x, x, CNT.z, CNT.x, CNT.y, CNT.z)
    top = V.gate("hex", 2, 2, -2, 2, 5, 10)
    bot = V.gate("hex", -2, -2, -2, 2, 5, 10)
    ox = V.region("hex", -2, 2, -2, 2, grid.gridz[0], grid.gridz[grid.nz - 1])
    ox.eps = 3.9
    p = V.interface3D(grid, top, bot, ox)
    V.dope_reservoir(grid, p, CNT, 5e-3, np.array([grid.xmin, grid.xmax, grid.ymin, grid.ymax, 0, 5]))
    V.dope_reservoir(grid, p, CNT, 5e-3, np.array([grid.xmin, grid.xmax, grid.ymin, grid.ymax, 10, 15]))
    p.modespace = "yes"
    CNT.Nmodes = 2
    CNT.mu2 = -0.1
    vgs = [round(0.1 * i, 3) for i in range(6)] if quick else [round(0.05 * i, 3) for i in range(11)]
    cur, iters = [], []
    for k, vg in enumerate(vgs):
        top.Ef = bot.Ef = -vg
        V.set_gate(p, top)
        V.set_gate(p, bot)
        if k == 0:
            p.normpoisson = 1e-3
            V.solve_init(grid, p, CNT)
        p.normpoisson = 1e-1
        p.normd = 5e-2
        n0 = solves()
        V.solve_self_consistent(grid, p, CNT)
        iters.append(solves() - n0)
        cur.append(float(CNT.current()))
    out = dev_result(CNT, "last bias", nch=2 * CNT.Nmodes)
    out["series"]["Id(Vgs)"] = cur
    out["x"]["Id(Vgs)"] = vgs
    out["info"] = {"SCF iterations per bias": iters}
    I = np.abs(np.array(cur))
    rise = float(max(0.0, np.max(I[:-1] - I[1:]))) / max(float(np.max(I)), 1e-300)
    out["laws"]["Id rises with Vgs"] = law(rise, 1e-6, "an n-FET's current grows with the gate voltage")
    dlog = np.diff(np.log10(np.maximum(I, 1e-300)))
    ss = [1e3 * (vgs[i + 1] - vgs[i]) / dlog[i] for i in range(len(dlog)) if dlog[i] > 0]
    ssmin = min(ss) if ss else float("inf")
    out["info"]["min SS (mV/dec)"] = ssmin
    # thermionic limit (kT/q) ln10 = 59.6 mV/dec at 300 K, 3% for the
    # finite Vgs step and the discrete energy grid
    out["laws"]["SS >= 60 mV/dec"] = law(max(0.0, 57.8 - ssmin), 1e-9,
                                         "no subthreshold swing below the thermionic limit")
    return out


def tut_graphene_idvds(quick):
    """demo/tutorial_idvds_graphene.py: output characteristic Id(Vds) of
    a double-gate graphene FET, 2D Poisson + NEGF summed over ky.
    Reduced (the original is an MPI job): full 30 nm with 32 ky and Vds
    0.1..0.5 step 0.1; quick 15 nm with 16 ky at Vds = 0.2.  Gate
    positions scale with the length.  No MPI (MPI_kt left "no")."""
    V = load()
    L, Nky = (15.0, 16) if quick else (30.0, 32)
    vds = [0.2] if quick else [0.1, 0.2, 0.3, 0.4, 0.5]
    xg = V.nonuniformgrid(np.array([-10, 1, 0, 0.05, 10, 1]))
    F = V.graphene(L)
    acc = 0.144
    kF = 2 * np.pi / (3 * np.sqrt(3) * acc)
    dk = 2.0 / (Nky * 0.5)
    F.kmax, F.kmin, F.dk = kF + dk * Nky * 0.5, kF - dk * Nky * 0.5, dk
    grid = V.grid2D(xg, F.y, F.x, F.y)
    ox = V.region("hex", grid.xmin, grid.xmax, grid.ymin, grid.ymax)
    ox.eps = 3.9
    g1, g2 = 0.3 * L, 0.7 * L
    top = V.gate("hex", grid.xmax, grid.xmax, g1, g2)
    bot = V.gate("hex", grid.xmin, grid.xmin, g1, g2)
    top.Ef = bot.Ef = -0.1
    p = V.interface2D(grid, ox, top, bot)
    V.dope_reservoir(grid, p, F, 5e-3, np.array([-1, 1, grid.ymin, g1]))
    V.dope_reservoir(grid, p, F, 5e-3, np.array([-1, 1, g2, grid.ymax]))
    V.solve_init(grid, p, F)
    p.underel = 0.1
    cur, iters = [], []
    for v in vds:
        F.mu2 = -v
        p.normpoisson = 1e-1
        p.normd = 5e-3
        n0 = solves()
        V.solve_self_consistent(grid, p, F)
        iters.append(solves() - n0)
        cur.append(float(F.current()))
    k = ne_of(F.E)
    out = {"series": {"Id(Vds)": cur, "T (last bias)": lst(np.asarray(F.T)[:k]),
                      "charge (last bias)": lst(F.charge), "Phi on channel": lst(p.Phi[grid.swap])},
           "x": {"Id(Vds)": vds, "T (last bias)": lst(np.asarray(F.E)[:k])},
           "scalars": {}, "laws": {},
           "info": {"NEGF solves (one per ky) per bias": iters}}
    T = np.asarray(F.T)[:k]
    out["laws"]["T >= 0"] = law(max(0.0, -float(np.min(T))) / max(1.0, float(np.max(T))), 1e-6,
                                "transmission is never negative")
    I = np.array(cur)
    if len(I) > 1:
        out["laws"]["Id rises with Vds"] = law(max(0.0, float(np.max(I[:-1] - I[1:]))) / float(np.max(np.abs(I))),
                                               1e-6, "current grows with the drain bias")
    return out


# ======================================================================
# R2 -- the regression tests in test/, against their stored references
# ======================================================================

def reg_gnr_fast_ohmic(quick):
    """test/script_GNR_auto.py: the same n=2 ribbon solved by two
    implementations -- nanoribbon (GNR_charge_T, the GNR path) and
    nanoribbon_fast_ohmic (a Hamiltonian built in Python, the H path) --
    must give the same charge (< 1e-3) and energy grid.  Unchanged
    (ctest attribute dropped: unused)."""
    V = load()
    H = V.nanoribbon_fast_ohmic(2, 2)
    H.Eupper, H.Elower, H.mu1, H.mu2, H.dE = 3, 0, 1.8, 1.8, 1e-2
    H.charge_T()
    G = V.nanoribbon(2, 1)
    G.Eupper, G.eta, G.Elower, G.mu1, G.mu2, G.dE = 3, 1e-5, 0, 1.8, 1.8, 1e-2
    G.charge_T()
    out = merge(dev_result(G, "nanoribbon", nch=G.n), dev_result(H, "fast_ohmic", nch=H.n))
    Np = np.size(G.charge)
    out["refs"] = {
        "charge: nanoribbon vs fast_ohmic": {"err": float(np.max(np.abs(G.charge - H.charge[:Np]))),
                                             "tol": 1e-3, "what": "two implementations agree (original test)"},
        "E grid: nanoribbon vs fast_ohmic": {"err": float(np.max(np.abs(G.E - H.E[:np.size(G.E)]))),
                                             "tol": 1e-3, "what": "same energy grid (original test)"}}
    return out


def _graphene_k(N, Efield_per_nm, Emin, Emax, dE):
    """test/test_demetrio_graphene_auto.py: a graphene strip of 4N slices
    summed over 100 transverse wave vectors k in [8, 9] nm^-1, one
    Hamiltonian NEGF per k (mu = 0.3)."""
    V = load()
    slices = 4 * N
    thop, delta = -2.7, 0.144 * np.sqrt(3)
    kmin, kmax, Np = 8.0, 9.0, 100
    dk = (kmax - kmin) / Np
    G = V.nanoribbon(1, N * 2.5 * 0.144 + (N - 1) * 0.144 * 0.5 + 2 * 0.144)
    zzz = G.z[0:-2]
    Efield = Efield_per_nm * zzz[-1]
    Ne = int(V.NEmax)
    Z = np.zeros((Ne, Np))
    h = np.zeros((2 * slices, 3), dtype=complex)
    h[0][0] = 1
    for i in range(1, slices + 1):
        h[i][0] = i
        h[i][1] = i
    kk = 1
    for ii in range(slices + 1, 2 * slices):
        if ii % 2 == 1:
            h[ii][0], h[ii][1], h[ii][2] = kk, kk + 1, thop
        kk += 1
    i, k = 0, kmin
    H = None
    while k < kmax * 0.99:
        flaggo, kk = 0, 1
        for ii in range(slices + 1, 2 * slices):
            if ii % 2 == 0:
                h[ii][0], h[ii][1] = kk, kk + 1
                if flaggo % 2 == 0:
                    h[ii][2] = thop + thop * np.exp(k * delta * 1j)
                else:
                    h[ii][2] = thop + thop * np.exp(-k * delta * 1j)
                flaggo += 1
            kk += 1
        H = V.Hamiltonian(1, slices)
        H.z = zzz
        H.Eupper, H.Elower = Emax, Emin
        H.H = h
        H.mu1 = H.mu2 = 0.3
        H.dE = dE
        H.Phi = Efield / H.z[-1] * H.z
        H.charge_T()
        Z[:, i] = H.T
        i += 1
        k += dk
    T = np.sum(Z, 1) * 2 * dk / (2 * np.pi) * 1e3
    return H, T


def reg_graphene(quick):
    """test/test_demetrio_graphene_auto.py, both tests: transmission of a
    graphene strip summed over k, at zero field (N=5, vs test/g0.dat,
    |sum T - sum ref| < 400) and with a field (N=44, vs test/g.dat,
    < 500).  Unchanged."""
    out = {"series": {}, "x": {}, "scalars": {}, "laws": {}, "refs": {}}
    for name, args, ref, tol in [("zero field", (5, 0.0, -0.2, 0.2002, 2e-4), "g0.dat", 4e2),
                                 ("field", (44, -0.005, -0.202, 0.2, 2e-3), "g.dat", 5e2)]:
        H, T = _graphene_k(*args)
        k = ne_of(H.E)
        G = np.loadtxt(os.path.join(TEST, ref))
        out["series"]["T " + name] = lst(T[:k])
        out["x"]["T " + name] = lst(np.asarray(H.E)[:k])
        out["refs"]["sum T, %s (%s)" % (name, ref)] = {
            "err": float(abs(np.sum(T) - np.sum(G[:, 1]))), "tol": tol,
            "what": "integrated transmission vs stored result (original test)"}
        m = min(k, len(G))
        out.setdefault("info", {})["max |T - ref| / peak, " + name] = float(
            np.max(np.abs(T[:m] - G[:m, 1])) / max(1.0, float(np.max(np.abs(G[:, 1])))))
        out["laws"]["T >= 0, " + name] = law(max(0.0, -float(np.min(T[:k]))), 1e-6, "transmission is never negative")
    return out


def reg_bilayer(quick):
    """test/test_bilayer_class_auto.py, all four tests: charge of a 10 nm
    bilayer graphene flake summed over 32 k, for Phi = -0.2, +0.5, a
    linear ramp and an alternating +-0.1 between layers, against the
    stored test/carica.* (tol 1e-5, 1e-5, 5e-5, 5e-5).  Unchanged."""
    V = load()
    out = {"series": {}, "scalars": {}, "laws": {}, "refs": {}}
    cases = [("-0.2", 1e-5, (0.633527, -0.433527), "carica.-0.2", 1e-5),
             ("+0.5", 1e-5, (0.258527, -0.933527), "carica.0.5", 1e-5),
             ("ramp", 1e-8, (0.433527, -0.530727), "carica.lin", 5e-5),
             ("+-0.1", 1e-5, (0.533527, -0.533527), "carica.plusminus", 5e-5)]
    for name, eta, (up, lo), ref, tol in cases:
        B = V.bilayer_graphene(10)
        B.dE, B.eta = 1e-3, eta
        B.kmax, B.kmin = 12.5958, 5.39722
        B.dk = (B.kmax - B.kmin) / 32.0
        B.Eupper, B.Elower = up, lo
        if name == "-0.2":
            B.Phi = -0.2 * np.ones(np.size(B.Phi))
        elif name == "+0.5":
            B.Phi = 0.5 * np.ones(np.size(B.Phi))
        elif name == "ramp":
            B.Phi = B.y * 1e-2
        else:
            i = np.arange(np.size(B.Phi))
            B.Phi[i % 2 == 0] = -0.1
            B.Phi[i % 2 == 1] = 0.1
            B.Ei = -B.Phi
        B.charge_T()
        c = B.charge * 0.144 * np.sqrt(3) * 1e-9
        out["series"]["charge, Phi " + name] = lst(c)
        out["refs"]["charge, Phi %s (%s)" % (name, ref)] = ref_maxabs(
            c, np.loadtxt(os.path.join(TEST, ref)), tol, "vs stored result (original test)")
        out["laws"]["finite, Phi " + name] = law(int(np.sum(~np.isfinite(c))), 0.5, "no NaN/inf")
    return out


def _snw(a, eta=1e-7):
    V = load()
    x, y, z = V.atoms_coordinates_nanowire(np.array(a))
    r = V.create_H_from_xyz(np.asarray(x), np.asarray(y), np.asarray(z), 10, V.onsite_Si, V.thop_Si, 3, 4)
    HH, n, Nc = r[0], r[1], r[2]
    S = V.Hamiltonian(n, Nc)
    S.H = HH
    S.Elower, S.Eupper, S.dE, S.eta = 0, 10, 0.1, eta
    S.charge_T()
    return S


def reg_snw(quick):
    """test/test_SNW.py: sp3d5s* silicon nanowires (10 orbitals per atom,
    surface passivation) through the Hamiltonian path, E in [0, 10] step
    0.1, eta 1e-7, transmission vs the stored test/transmission-test-Si*.dat.
    Test 1 always (tol 5e-5); tests 2 and 3 (wider wires, tol 1e-1 and
    5e-5) only in the full run.  Updated to the current
    create_H_from_xyz, which returns six values instead of three."""
    tests = [("wire 1", [5.431, 0, 1, 3.85, 15], "transmission-test-Si.dat", 5e-5)]
    if not quick:
        tests += [("wire 2", [5.431, 0, 0, 7.85, 20], "transmission-test-Si-2.dat", 1e-1),
                  ("wire 3", [5.431, 0, 1, 7.85, 20], "transmission-test-Si-1.dat", 5e-5)]
    out = {"series": {}, "x": {}, "scalars": {}, "laws": {}, "refs": {}, "info": {}}
    for name, a, ref, tol in tests:
        S = _snw(a)
        r = dev_result(S, name)
        out = merge(out, r)
        b = np.loadtxt(os.path.join(TEST, ref))
        out["refs"]["T %s (%s)" % (name, ref)] = ref_maxabs(S.T, b[:, 1], tol, "vs stored result (original test)")
        out["info"]["%s: n x Nc" % name] = "%d x %d" % (S.n, S.Nc)
    return out


def reg_zincblend(quick):
    """test/script_test_sp3d5ss_auto.py: sp3d5s* Si, Ge and InAs wires
    built by the Zincblend class, flat band, E in [0, 10] step 0.1,
    eta 1e-8, vs the stored test/transmission-test-{Si,Ge,InAs}.dat
    (tol 1e-5).  Python 3 port of the script; skipped if the Zincblend
    class itself does not run under Python 3."""
    V = load()
    params = {"Si": 3.85, "InAs": 4.29, "Ge": 4.00}
    out = {"series": {}, "x": {}, "scalars": {}, "laws": {}, "refs": {}}
    for mat in ("Si", "Ge", "InAs"):
        try:
            H = V.Zincblend(mat, 0, 1, params[mat], 0.0)
        except Exception as e:      # library-side, identical on both backends
            raise Skip("Zincblend(%r) fails under this Python: %s: %s" % (mat, type(e).__name__, e))
        H.Elower, H.Eupper, H.dE, H.eta = 0, 10, 0.1, 1e-8
        H.mu1 = H.mu2 = 0.0
        n, Nc = H.n, H.Nc
        length = H.z[(Nc - 1) * n + (n - 1)]
        Phi = np.zeros(n * Nc)
        for i in range(4, Nc - 4):
            for j in range(n):
                k = i * n + j
                Phi[k] = H.mu1 * (1 - 2 * ((H.z[k] - H.z[4 * n]) / (length - 2 * H.z[4 * n])))
        H.Phi = Phi
        H.charge_T()
        out = merge(out, dev_result(H, mat))
        a = np.loadtxt(os.path.join(TEST, "transmission-test-%s.dat" % mat))
        out["refs"]["T %s" % mat] = ref_maxabs(H.T, a[:, 1], 1e-5, "vs stored result (original test)")
    return out


def reg_hole(quick):
    """test/test_hole.py: a 5-wide ribbon with one atom removed, built two
    ways -- from ribbon5_hole.xyz (the atom absent) and from ribbon5.xyz
    with that atom's three bonds cut -- must have the same transmission
    (except exactly at E = 0, where the cut-out atom of the second is an
    isolated state).  Updated to the current create_H_from_xyz signature
    (onsite argument, six return values)."""
    V = load()
    x, y, z = V.get_xyz_from_file(os.path.join(TEST, "ribbon5_hole.xyz"))
    r = V.create_H_from_xyz(x, y, z, 1, 0.0, -2.7, 1.45, 3)
    A = V.Hamiltonian(r[1], r[2])
    A.H = r[0]
    A.Eupper, A.Elower, A.dE = 1.5, -1.5, 0.1
    A.charge_T()
    x, y, z = V.get_xyz_from_file(os.path.join(TEST, "ribbon5.xyz"))
    r = V.create_H_from_xyz(x, y, z, 1, 0.0, -2.7, 1.45, 3)
    Hold = r[0]
    for a, b in ((28, 32), (28, 33), (23, 28)):
        ind = np.nonzero((np.real(Hold[:, 0]) == a) & (np.real(Hold[:, 1]) == b))[0]
        Hold[ind, 2] = 0
    B = V.Hamiltonian(r[1], r[2])
    B.H = Hold
    B.Eupper, B.Elower, B.dE = 1.5, -1.5, 0.1
    B.charge_T()
    out = merge(dev_result(A, "hole (xyz)", equilibrium=True),
                dev_result(B, "hole (bonds cut)", equilibrium=True))
    k = min(ne_of(A.E), ne_of(B.E))
    E = np.asarray(A.E)[:k]
    keep = np.abs(E) > 1e-6
    d = np.abs(np.asarray(A.T)[:k] - np.asarray(B.T)[:k])[keep]
    out["refs"] = {"T: xyz hole vs bonds cut": {
        "err": float(np.max(d)) / max(1.0, float(np.max(np.asarray(A.T)[:k]))), "tol": 1e-6,
        "what": "two constructions of the same defect agree (E = 0 excluded)"}}
    return out


# ======================================================================
# P1 -- device physics every correct solver must reproduce
# ======================================================================

GRID = [-1.5025, 1.5025, 0.005]     # symmetric, no point exactly at E = 0


def _flat(kind, size, Phi=None, eta=None):
    V = load()
    if kind == "cnt":
        d = V.nanotube(*size)
    elif kind == "gnr":
        d = V.nanoribbon(*size)
    else:
        from GNR import GNR
        d = V.Hamiltonian(*size)
        d.H = GNR(*size)
        d.eta = 1e-5
    d.Elower, d.Eupper, d.dE = GRID
    if eta is not None:
        d.eta = eta
    if Phi is not None:
        d.Phi = Phi(d)
    d.charge_T()
    return d


def _ET(d):
    k = ne_of(d.E)
    return np.asarray(d.E, float)[:k], np.asarray(d.T, float)[:k]


def phys_ballistic(quick):
    """Perfect, flat-band wires (Phi = 0): the transmission is the number
    of open modes, an integer, and the nearest-neighbour pz lattice is
    bipartite, so T(E) = T(-E).  CNT (13,0), GNR n=7 and the 5-wide
    ribbon through the Hamiltonian path, at eta = 1e-8.  Points within 25 meV of a
    subband edge (where T jumps) are excluded from the integer test."""
    # (25 meV: the T4 run showed ~3e-3 deviations right next to the edges
    # on both backends alike, i.e. eta broadening, not the GPU)
    out = {"series": {}, "x": {}, "scalars": {}, "laws": {}}
    L = 3.0 if quick else 6.0
    for kind, size in (("cnt", (13, L)), ("gnr", (7, L)), ("hamiltonian", (5, 24 if quick else 48))):
        # eta = 1e-8: the broadening is also an absorption inside the
        # device, which lowers T by ~ eta * length / velocity -- at the
        # default 1e-5 that is 1e-3 near the band edges (T4 run), and the
        # "T is an integer" law only holds as eta -> 0.
        d = _flat(kind, size, eta=1e-8)
        E, T = _ET(d)
        out = merge(out, dev_result(d, kind, nch=d.n, equilibrium=True))
        Tr = np.round(T)
        jump = np.zeros(len(T), bool)
        step = np.nonzero(np.diff(Tr) != 0)[0]
        for s in step:
            jump[max(0, s - 5):s + 7] = True
        out["laws"][kind + ": T integer"] = law(float(np.max(np.abs(T - Tr)[~jump])) if (~jump).any() else 0,
                                                1e-4, "flat band: T counts open modes")
        out["laws"][kind + ": T(E) = T(-E)"] = law(float(np.max(np.abs(T - T[::-1]))) / max(1.0, float(np.max(T))),
                                                   1e-6, "electron-hole symmetry of the bipartite lattice")
    return out


def shifted_err(E0, T0, E1, T1, s):
    """max |T1(E) - T0(E - s)| over the energies both grids hold."""
    dE = E0[1] - E0[0]
    j = np.round((E1 - s - E0[0]) / dE).astype(int)
    ok = (j >= 0) & (j < len(E0))
    ok &= np.abs(E0[np.clip(j, 0, len(E0) - 1)] - (E1 - s)) < 1e-3 * dE
    return float(np.max(np.abs(T1[ok] - T0[j[ok]]))) if ok.any() else float("inf")


def band_sign(kind, size):
    """+1 if a positive Phi moves the bands up in energy, -1 if down."""
    E0, T0 = _ET(_flat(kind, size))
    E1, T1 = _ET(_flat(kind, size, lambda d: 0.1 * np.ones(np.size(d.Phi))))
    return 1 if shifted_err(E0, T0, E1, T1, 0.1) <= shifted_err(E0, T0, E1, T1, -0.1) else -1


def phys_shift_reciprocity(quick):
    """Two exact symmetries, on a CNT (13,0) and a GNR n=6.
    Shift: a uniform potential Phi = 0.1 moves every band rigidly, so
    T(E) is the flat-band T(E) moved by 0.1 eV (20 grid steps).
    Reciprocity: mirroring an asymmetric potential (ramp plus off-centre
    bump) end to end leaves T(E) unchanged (T_LR = T_RL)."""
    out = {"series": {}, "x": {}, "scalars": {}, "laws": {}, "info": {}}
    L = 4.0 if quick else 8.0
    for kind, size in (("cnt", (13, L)), ("gnr", (6, L))):
        d0 = _flat(kind, size)
        E0, T0 = _ET(d0)
        d1 = _flat(kind, size, lambda d: 0.1 * np.ones(np.size(d.Phi)))
        E1, T1 = _ET(d1)
        # bands move up by Phi or down, depending on the sign convention;
        # score the one the code uses
        up, dn = shifted_err(E0, T0, E1, T1, 0.1), shifted_err(E0, T0, E1, T1, -0.1)
        out["laws"][kind + ": rigid shift"] = law(min(up, dn) / max(1.0, float(np.max(T0))), 1e-6,
                                                  "uniform potential = rigid band shift")
        out["info"][kind + ": bands move"] = "up" if up <= dn else "down"
        zf = lambda d: (d.z - np.min(d.z)) / max(np.ptp(d.z), 1e-12)
        prof = lambda u: -0.15 * u + 0.25 * np.exp(-((u - 0.3) / 0.12) ** 2)
        da = _flat(kind, size, lambda d: prof(zf(d)))
        db = _flat(kind, size, lambda d: prof(1 - zf(d)))
        Ea, Ta = _ET(da)
        Eb, Tb = _ET(db)
        out["laws"][kind + ": reciprocity"] = law(float(np.max(np.abs(Ta - Tb))) / max(1.0, float(np.max(Ta))),
                                                  1e-6, "mirrored potential, same T(E)")
        out["series"][kind + ": T shifted"] = lst(T1)
        out["x"][kind + ": T shifted"] = lst(E1)
        out["series"][kind + ": T asymmetric"] = lst(Ta)
        out["x"][kind + ": T asymmetric"] = lst(Ea)
        out["series"][kind + ": T mirrored"] = lst(Tb)
        out["x"][kind + ": T mirrored"] = lst(Eb)
    return out


def phys_neutral(quick):
    """Charge neutrality: at flat band with mu1 = mu2 = 0 the intrinsic
    CNT and GNR hold no net charge (electron-hole symmetry), while a
    potential of -0.2 V does induce charge.  The neutral charge must be
    tiny compared with the induced one."""
    out = {"series": {}, "scalars": {}, "laws": {}}
    for kind, size in (("cnt", (13, 3.0)), ("gnr", (6, 3.0))):
        d0 = _flat(kind, size)
        d1 = _flat(kind, size, lambda d: -0.2 * np.ones(np.size(d.Phi)))
        c0, c1 = np.asarray(d0.charge).ravel(), np.asarray(d1.charge).ravel()
        out["series"][kind + ": charge, Phi=-0.2"] = lst(c1)
        out["laws"][kind + ": neutral at flat band"] = law(
            float(np.max(np.abs(c0))) / max(float(np.max(np.abs(c1))), 1e-300), 1e-3,
            "no net charge at the charge-neutrality point")
        out["laws"][kind + ": gating induces charge"] = law(
            0.0 if np.max(np.abs(c1)) > 0 else 1.0, 0.5, "a potential of -0.2 V induces charge")
    return out


def phys_iv(quick):
    """Output characteristics of a CNT (13,0) and a GNR n=6 carrying
    electrons in the conduction band (mu1 = Eg/2 + 0.2 eV, mu2 = mu1 -
    Vds) through a fixed Gaussian barrier in the middle of the channel:
    I(0) = 0 exactly; |I| grows monotonically with Vds (f1 - f2 grows at
    every energy); and I falls as the barrier is raised, at Vds = 0.2.
    The barrier's sign follows the code's own potential convention
    (band_sign), so it is a barrier for electrons either way."""
    V = load()
    out = {"series": {}, "x": {}, "scalars": {}, "laws": {}, "info": {}}
    L = 6.0 if quick else 10.0
    vds = [0.0, 0.1, 0.2, 0.3] if quick else [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    hs = [0.1, 0.2, 0.3] if quick else [0.1, 0.2, 0.3, 0.4]
    for kind, size in (("cnt", (13, L)), ("gnr", (6, L))):
        sg = band_sign(kind, (size[0], 2.0))
        make = V.nanotube if kind == "cnt" else V.nanoribbon

        def run(h, v):
            d = make(*size)
            mu1 = 0.5 * float(d.gap()) + 0.2
            d.Elower, d.Eupper, d.dE = mu1 - 0.8025, mu1 + 0.4025, 0.005
            u = (d.z - np.min(d.z)) / max(np.ptp(d.z), 1e-12)
            d.Phi = sg * h * np.exp(-((u - 0.5) / 0.15) ** 2)
            d.mu1, d.mu2 = mu1, mu1 - v
            d.charge_T()
            return float(d.current())
        cur = [run(0.2, v) for v in vds]
        I = np.abs(np.array(cur))
        out["series"][kind + ": I(Vds)"] = cur
        out["x"][kind + ": I(Vds)"] = vds
        out["laws"][kind + ": I(0) = 0"] = law(I[0] / max(float(np.max(I)), 1e-300), 1e-9,
                                               "no current without bias")
        out["laws"][kind + ": I grows with Vds"] = law(
            max(0.0, float(np.max(I[:-1] - I[1:]))) / max(float(np.max(I)), 1e-300), 1e-9,
            "f1 - f2 grows at every energy")
        cb = [run(h, 0.2) for h in hs]
        Ib = np.abs(np.array(cb))
        out["series"][kind + ": I(barrier)"] = cb
        out["x"][kind + ": I(barrier)"] = hs
        out["laws"][kind + ": I falls with barrier"] = law(
            max(0.0, float(np.max(Ib[1:] - Ib[:-1]))) / max(float(np.max(Ib)), 1e-300), 1e-6,
            "a higher barrier passes less current")
        out["info"][kind + ": bands move with +Phi"] = "up" if sg > 0 else "down"
    return out


def reg_zincblend_field(quick):
    """test/script_test_sp3d5ss_field_auto.py: the Zincblend wires of
    reg_zincblend with a +-V/2 bias (V = Egap + 0.3 eV) across a linear
    potential drop, E from the band edge Ei to Ei + 0.1 eV in 1 meV
    steps, eta 1e-8, vs the stored test/transmission-test-{Si,InAs,Ge}-
    field.dat (tol 1e-5).  Python 3 port; the potential is built exactly
    as the script does (linear drop on the active slices, mu1 / mu2 on the
    contact slices)."""
    V = load()
    # material: (edge, Egap, Ei) as in the script
    params = {"Si": (3.85, 6.00, 1.00), "InAs": (4.29, 4.465433, 1.00), "Ge": (4.00, 5.2, 1.07)}
    out = {"series": {}, "x": {}, "scalars": {}, "laws": {}, "refs": {}}
    for mat in ("Si", "InAs", "Ge"):
        edge, egap, ei = params[mat]
        Voltage = egap + 0.3
        try:
            H = V.Zincblend(mat, 0, 1, edge, Voltage / 1.0)
        except Exception as e:
            raise Skip("Zincblend(%r) fails under this Python: %s: %s" % (mat, type(e).__name__, e))
        H.Elower, H.Eupper, H.dE, H.eta = ei, ei + 0.1, 0.001, 1e-8
        v = Voltage / 2
        H.mu1, H.mu2 = v, -v
        n, Nc = H.n, H.Nc
        H.Phi = H.mu1 * (1 - 2 * ((H.z - H.a0) / H.L))
        H.Phi = V.ciccione(H.Phi, n, Nc, H.z, H.a0)
        H.z = V.ciccione(H.z, n, Nc, H.z, H.a0)
        for i in range(4 * n):
            H.Phi[i] = H.Phi[0]
        for i in range(Nc - 4, Nc):
            for j in range(n):
                H.Phi[i * n + j] = H.mu2
        for i in range(4, Nc - 4):
            for j in range(n):
                H.Phi[i * n + j] = H.mu1 * (1 - 2 * ((H.z[i * n] - 2 * H.a0) / (H.L - 2 * H.a0)))
        H.charge_T()
        out = merge(out, dev_result(H, mat))
        a = np.loadtxt(os.path.join(TEST, "transmission-test-%s-field.dat" % mat))
        out["refs"]["T %s field" % mat] = ref_maxabs(H.T, a[:, 1], 1e-5, "vs stored result (original test)")
    return out


def reg_wire2(quick):
    """test/test_wire2.py: a 10.5 A long, 3.85 A wide Si wire (sp3d5s*,
    10 orbitals per atom) through the Hamiltonian path, E in [0, 10] step
    0.1.  The original sets eta = 1e-12, which no solver resolves (the
    broadening is below round-off of the energy grid); eta 1e-8 here, as in
    the other wire tests.  No stored reference for this length: judged by
    CPU-vs-GPU agreement and the transmission laws."""
    V = load()
    x, y, z = V.atoms_coordinates_nanowire(np.array([5.43, 0, 1, 3.85, 10.5]))
    r = V.create_H_from_xyz(np.asarray(x), np.asarray(y), np.asarray(z), 10, V.onsite_Si, V.thop_Si, 3, 4)
    S = V.Hamiltonian(r[1], r[2])
    S.H = r[0]
    S.Elower, S.Eupper, S.dE, S.eta = 0, 10, 0.1, 1e-8
    S.charge_T()
    out = dev_result(S, "wire", equilibrium=True)
    out["info"] = {"n x Nc": "%d x %d" % (S.n, S.Nc)}
    return out


def reg_scf_graphene(quick):
    """test/test_selfconsistent_graphene.py: double-gate graphene flake
    (30 nm), self-consistent over ky in [7, 10] (Nk = 30; quick 10), top
    gate -0.6 V, doped reservoirs.  The original starts from a stored
    Phi.MPI that is not in the repository; the starting potential is the
    Poisson solution instead.  MPI_kt left off.  Poisson 1e-1, normd 5e-2
    as written."""
    V = load()
    xg = V.nonuniformgrid(np.array([-1, 0.2, 0, 0.05, 1, 0.2]))
    F = V.graphene(30)
    Nk = 10 if quick else 30
    F.kmin, F.kmax = 7, 10
    F.dk = float(F.kmax - F.kmin) / Nk
    grid = V.grid2D(xg, F.y, F.x, F.y)
    top = V.gate("hex", grid.xmax, grid.xmax, 10, 20)
    bot = V.gate("hex", grid.xmin, grid.xmin, 10, 20)
    top.Ef, bot.Ef = -0.6, 0
    ox = V.region("hex", grid.xmin, grid.xmax, grid.ymin, grid.ymax)
    ox.eps = 3.9
    p = V.interface2D(grid, top, bot, ox)
    p.normpoisson = 1e-3
    V.solve_Poisson(grid, p)
    V.dope_reservoir(grid, p, F, 5e-3, np.array([-1, 1, grid.ymin, 10]))
    V.dope_reservoir(grid, p, F, 5e-3, np.array([-1, 1, 20, grid.ymax]))
    p.normpoisson = 1e-1
    p.normd = 5e-2
    V.solve_self_consistent(grid, p, F)
    out = dev_result(F)
    out["series"]["Phi on channel"] = lst(p.Phi[grid.swap])
    out["info"] = {"SCF iterations": solves(), "ky points": Nk}
    return out


def tut_gnrphonon(quick):
    """demo/tutorial_GNRphonon.py: electron and phonon dispersions of a
    10-dimer ribbon and the acoustic / optical absorption and emission
    rates.  Pure CPU code with no NEGF solve, so the GPU build must leave
    it untouched: the results of a GPU-enabled process must equal the
    CPU-only ones exactly."""
    V = load()
    a = V.GNRphonon(10)
    a.electron_GNR()
    a.phonon_GNR()
    a.rateACABS()
    a.rateACEM()
    a.rateOPTABS()
    a.rateOPTEM()
    a.phonon_graphene()
    ser = {"electron bands": a.energyE, "phonon branches": a.energyP2D, "AC absorption": a.rateAA,
           "AC emission": a.rateAE, "optical absorption": a.rateOA, "optical emission": a.rateOE,
           "graphene phonons": a.Egraphene}
    out = {"series": {k: lst(v) for k, v in ser.items()}, "scalars": {}, "laws": {}, "cpu_only": True}
    allv = np.concatenate([np.asarray(v, float).ravel() for v in ser.values()])
    out["laws"]["finite"] = law(int(np.sum(~np.isfinite(allv))), 0.5, "no NaN/inf in any dispersion or rate")
    out["laws"]["rates >= 0"] = law(max(0.0, -float(min(np.min(a.rateAA), np.min(a.rateAE),
                                                           np.min(a.rateOA), np.min(a.rateOE)))),
                                    1e-12, "scattering rates are never negative")
    return out


SCENARIOS = {f.__name__: f for f in [
    tut_cnt, tut_gnr, tut_hamilton, tut_poisson_cnt, tut_scf_cnt, tut_scf_cnt_tight,
    tut_scf_gnr_schottky,
    tut_idvgs, tut_graphene_idvds,
    reg_gnr_fast_ohmic, reg_graphene, reg_bilayer, reg_snw, reg_zincblend, reg_hole,
    reg_zincblend_field, reg_wire2, reg_scf_graphene, tut_gnrphonon,
    phys_ballistic, phys_shift_reciprocity, phys_neutral, phys_iv]}
