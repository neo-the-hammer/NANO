"""Draw a simulated device from the geometry capture.py recorded.

render(geom, title, plt) returns one matplotlib figure per device:

  (a) side view: transport direction across, the gate direction up; the
      dielectric regions (shaded, labelled with eps), the doped reservoirs
      (hatched, with the molar fraction), the gates (dark bars, with their
      Fermi level) and the atoms coloured by their charge
  (b) top view: transport direction across, the remaining direction up,
      atoms coloured by the electrostatic potential (only when the device
      has extent in that direction)
  (c) along the transport direction: the band shift -Phi and the charge,
      averaged over each slice of the device

The transport direction is the one in which the atoms extend furthest.
For a flat device (graphene, 2D materials) the side view takes its
vertical direction from the gates and the oxide.
"""

AX = ("x", "y", "z")
SPAN = {"x": ("xmin", "xmax"), "y": ("ymin", "ymax"), "z": ("zmin", "zmax")}
BOX = {2: ("x",), 4: ("x", "y"), 6: ("x", "y", "z")}


def _extent(v):
    return (max(v) - min(v)) if v else 0.0


def _span(obj, ax):
    lo, hi = SPAN[ax]
    if lo in obj and hi in obj:
        return obj[lo], obj[hi]
    return None


def _box_span(d, ax):
    """Span of a doped box along ax: from the bbox list or region bounds."""
    b = d.get("box")
    if b:
        axes = BOX.get(len(b))
        if axes and ax in axes:
            i = axes.index(ax)
            return b[2 * i], b[2 * i + 1]
        return None
    return _span(d, ax)


def axes_for(dev, geom):
    ext = {a: _extent(dev[a]) for a in AX}
    trans = max(AX, key=lambda a: ext[a])
    rest = [a for a in AX if a != trans]
    # vertical: the gate direction -- the axis of the thinnest gate (gates are planes)
    gate_ax = None
    for g in geom.get("gates", []):
        for a in rest:
            s = _span(g, a)
            if s and abs(s[1] - s[0]) < 1e-9:
                gate_ax = a
                break
        if gate_ax:
            break
    if gate_ax is None:
        gate_ax = max(rest, key=lambda a: ext[a])
    third = [a for a in rest if a != gate_ax][0]
    return trans, gate_ax, third


def _limits(vals, objs, ax, getter):
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    for o in objs:
        s = getter(o, ax)
        if s:
            lo, hi = min(lo, s[0]), max(hi, s[1])
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 0.04 * (hi - lo)
    return lo - pad, hi + pad


def _rect(a, x0, x1, y0, y1, **kw):
    from matplotlib.patches import Rectangle
    a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, **kw))


def _sym(v):
    m = max((abs(x) for x in v), default=0.0)
    return (-m, m) if m > 0 else (-1.0, 1.0)


def _profile(t, f, bins=60):
    lo, hi = min(t), max(t)
    if hi - lo <= 0:
        return [], []
    w = (hi - lo) / bins
    acc = [[0.0, 0] for _ in range(bins)]
    for x, v in zip(t, f):
        k = min(bins - 1, int((x - lo) / w))
        acc[k][0] += v
        acc[k][1] += 1
    xs = [lo + (k + 0.5) * w for k in range(bins) if acc[k][1]]
    ys = [acc[k][0] / acc[k][1] for k in range(bins) if acc[k][1]]
    return xs, ys


def render_one(dev, geom, title, plt):
    trans, up, third = axes_for(dev, geom)
    T, U, W = dev[trans], dev[up], dev[third]
    has_top = _extent(W) > 1e-6
    ncol = 3 if has_top else 2
    fig, ax = plt.subplots(1, ncol, figsize=(5.2 * ncol, 3.9), squeeze=False)
    ax = ax[0]
    fig.suptitle("%s -- %s (%d atoms%s)" % (title, dev["class"], dev["atoms"],
                                            ", every %d-th drawn" % dev["step"] if dev["step"] > 1 else ""),
                 x=0.01, ha="left")
    regions, gates, doping = geom.get("regions", []), geom.get("gates", []), geom.get("doping", [])
    xl = _limits(T, regions + gates, trans, _span)
    yl = _limits(U, regions + gates, up, _span)

    # (a) side view
    a = ax[0]
    for r in regions:
        sx, sy = _span(r, trans) or xl, _span(r, up) or yl
        _rect(a, sx[0], sx[1], sy[0], sy[1], facecolor="#e7e4d8", edgecolor="#c3c2b7", lw=0.6, zorder=0)
        a.text(sx[0], sy[1], " eps %.3g" % r.get("eps", 0), fontsize=6.5, color="#898781", va="top", zorder=1)
    for d in doping:
        sx, sy = _box_span(d, trans) or xl, _box_span(d, up) or yl
        _rect(a, sx[0], sx[1], sy[0], sy[1], facecolor="none", edgecolor="#3a9d5d", hatch="///", lw=0.8, zorder=2)
        a.text(0.5 * (sx[0] + sx[1]), sy[0], "doped %.3g" % d.get("fraction", 0), fontsize=6.5,
               color="#3a9d5d", ha="center", va="bottom", zorder=3)
    thick = 0.02 * (yl[1] - yl[0])
    for g in gates:
        sx, sy = _span(g, trans) or xl, _span(g, up) or yl
        y0, y1 = sy
        if y1 - y0 < thick:
            y0, y1 = 0.5 * (y0 + y1) - thick, 0.5 * (y0 + y1) + thick
        _rect(a, sx[0], sx[1], y0, y1, facecolor="#3b3a37", edgecolor="none", zorder=4)
        a.text(0.5 * (sx[0] + sx[1]), y1, "gate Ef %.2f" % g.get("Ef", 0), fontsize=6.5, ha="center",
               va="bottom", zorder=5)
    q = dev.get("charge")
    if q:
        lo, hi = _sym(q)
        sc = a.scatter(T, U, c=q, s=6, cmap="RdBu_r", vmin=lo, vmax=hi, zorder=6, linewidths=0)
        fig.colorbar(sc, ax=a, fraction=0.04, pad=0.02).set_label("charge per atom", fontsize=7)
    else:
        a.scatter(T, U, s=6, color="#2a78d6", zorder=6, linewidths=0)
    a.set_xlim(*xl)
    a.set_ylim(*yl)
    a.set_xlabel("%s (nm), transport" % trans)
    a.set_ylabel("%s (nm)" % up)
    a.set_title("(a) side view: gates, dielectrics, doping", fontsize=8.5)

    # (b) top view
    k = 1
    if has_top:
        b = ax[1]
        phi = dev.get("Phi")
        if phi:
            sc = b.scatter(T, W, c=[-v for v in phi], s=6, cmap="viridis", linewidths=0)
            fig.colorbar(sc, ax=b, fraction=0.04, pad=0.02).set_label("-Phi (eV)", fontsize=7)
        else:
            b.scatter(T, W, s=6, color="#2a78d6", linewidths=0)
        b.set_xlabel("%s (nm), transport" % trans)
        b.set_ylabel("%s (nm)" % third)
        b.set_title("(b) top view", fontsize=8.5)
        b.set_aspect("equal", adjustable="datalim")
        k = 2

    # (c) profiles along the transport direction
    c = ax[k]
    drawn = False
    if dev.get("Phi"):
        xs, ys = _profile(T, [-v for v in dev["Phi"]])
        c.plot(xs, ys, color="#eb6834", lw=1.6, label="-Phi (band shift)")
        c.set_ylabel("-Phi (eV)")
        drawn = True
    if q:
        c2 = c.twinx() if drawn else c
        xs, ys = _profile(T, q)
        c2.plot(xs, ys, color="#2a78d6", lw=1.2, ls="--", label="charge")
        c2.set_ylabel("charge per atom")
        drawn = True
    if drawn:
        for g in gates:
            s = _span(g, trans)
            if s:
                c.axvspan(s[0], s[1], color="#3b3a37", alpha=0.08, lw=0)
    c.set_xlabel("%s (nm), transport" % trans)
    c.set_title("(c) along the device (gates shaded)", fontsize=8.5)
    fig.tight_layout()
    return fig


def render(geom, title, plt):
    figs = []
    for dev in geom.get("devices", []):
        try:
            figs.append(render_one(dev, geom, title, plt))
        except Exception as e:  # a drawing problem must not cost the report
            print("   device drawing failed for %s: %s" % (title, e))
    return figs
