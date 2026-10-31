"""Record the geometry of the devices a worker process solves.

Recorder(V) wraps, in the NanoTCAD_ViDES module V,

  gate and region constructors   -> every gate / dielectric region made
  dope_reservoir                 -> every doped box and its molar fraction
  solve_self_consistent          -> the device of a Poisson-NEGF loop
  charge_T / mode_charge_T       -> every device solved directly

and serialize() returns plain JSON: the devices (atoms x, y, z in nm,
potential and charge per atom when they align with the atoms) and the
electrostatic set-up.  Large devices are subsampled to MAX_ATOMS atoms.
Devices without atom coordinates (a bare Hamiltonian, whose x, y, z stay
zero) are skipped.
"""

MAX_ATOMS = 6000
NUMERIC = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "Ef", "eps", "rho", "xc", "yc", "radius")
DEVICE_CLASSES = ("nanotube", "nanoribbon", "nanoribbon_fast_ohmic", "graphene", "bilayer_graphene",
                  "Hamiltonian", "Zincblend")


def _list(a):
    try:
        return [float(v) for v in list(a)]
    except TypeError:
        return None


def _flat(a):
    try:
        import numpy as np
        return [float(v) for v in np.asarray(a, dtype=float).ravel()]
    except Exception:
        return _list(a)


def _attrs(obj):
    d = {"geometry": str(getattr(obj, "geometry", "hex"))}
    for k in NUMERIC:
        v = getattr(obj, k, None)
        try:
            if v is not None:
                d[k] = float(v)
        except (TypeError, ValueError):
            pass
    return d


def device_geometry(d, name=None):
    """Atoms and per-atom fields of one device object, or None."""
    coords = {}
    for ax in ("x", "y", "z"):
        v = getattr(d, ax, None)
        coords[ax] = _flat(v) if v is not None else None
    n = max((len(v) for v in coords.values() if v), default=0)
    if n == 0:
        return None
    for ax in ("x", "y", "z"):
        if not coords[ax] or len(coords[ax]) != n:
            coords[ax] = [0.0] * n
    extent = max(max(v) - min(v) for v in coords.values())
    if extent <= 0:
        return None                          # no coordinates: nothing to draw
    fields = {}
    for f in ("Phi", "charge"):
        v = getattr(d, f, None)
        v = _flat(v) if v is not None else None
        if v and len(v) == n:
            fields[f] = v
    step = max(1, n // MAX_ATOMS)
    g = {"class": type(d).__name__, "name": name or type(d).__name__, "atoms": n, "step": step}
    for ax in ("x", "y", "z"):
        g[ax] = coords[ax][::step]
    for f, v in fields.items():
        g[f] = v[::step]
    return g


class Recorder:
    def __init__(self, V):
        self.V = V
        self.gates, self.regions, self.doping, self.devices = [], [], [], []
        self._wrap_init(V.gate, self.gates)
        self._wrap_init(V.region, self.regions)
        orig_dope = V.dope_reservoir

        def dope(grid, interface, channel, fraction, box, *a, **k):
            self.doping.append((box, fraction, type(grid).__name__))
            return orig_dope(grid, interface, channel, fraction, box, *a, **k)
        V.dope_reservoir = dope
        orig_scf = V.solve_self_consistent

        def scf(grid, interface, channel, *a, **k):
            r = orig_scf(grid, interface, channel, *a, **k)
            self._seen(channel, scf=True)
            return r
        V.solve_self_consistent = scf
        for cname in DEVICE_CLASSES:
            cls = getattr(V, cname, None)
            for meth in ("charge_T", "mode_charge_T"):
                f = getattr(cls, meth, None) if cls else None
                if f is None:
                    continue

                def wrapped(obj, _f=f):
                    r = _f(obj)
                    self._seen(obj)
                    return r
                setattr(cls, meth, wrapped)

    def _wrap_init(self, cls, store):
        orig = cls.__init__

        def init(obj, *a, **k):
            orig(obj, *a, **k)
            store.append(obj)
        cls.__init__ = init

    def _seen(self, obj, scf=False):
        for i, (o, s) in enumerate(self.devices):
            if o is obj:
                self.devices[i] = (o, s or scf)
                return
        self.devices.append((obj, scf))
        del self.devices[:-12]                  # the last few distinct devices are enough

    def serialize(self, name=""):
        devs = []
        seen_scf = any(s for _, s in self.devices)
        for obj, scf in self.devices:
            if seen_scf and not scf:
                continue                        # the internal per-ky / per-mode solves of a SCF device
            g = device_geometry(obj, name)
            if g:
                devs.append(g)
        devs = devs[-3:]
        dop = []
        for box, frac, gname in self.doping:
            b = _list(box)
            dop.append({"box": b, "fraction": float(frac)} if b else dict(_attrs(box), fraction=float(frac)))
        return {"devices": devs, "gates": [_attrs(g) for g in self.gates],
                "regions": [_attrs(r) for r in self.regions], "doping": dop}
