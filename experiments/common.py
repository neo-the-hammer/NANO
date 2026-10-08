"""Shared machinery for the CPU-vs-GPU experiments suite.

  - paths and the native process harness (build + run)
  - the device worker runner (one backend per subprocess)
  - Step / cost model / Progress: per-experiment estimates and live ETA
  - plot style and small numeric helpers

Nothing here needs NumPy; plotting needs matplotlib (imported lazily).
"""

import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
DEMO = os.path.join(ROOT, "demo")
NATIVE_SRC = os.path.join(HERE, "native", "vides_native.c")
RESULTS = os.path.join(HERE, "results")


# ======================================================================
# Native harness
# ======================================================================

def module_path():
    p = os.path.join(SRC, "NanoTCAD_ViDESmod.so")
    return p if os.path.exists(p) else None


def module_has_gpu():
    """True if src/NanoTCAD_ViDESmod.so was linked with the CUDA backend."""
    p = module_path()
    if not p:
        return False
    with open(p, "rb") as f:
        return b"cublasZgetrfBatched" in f.read()


def _run(cmd, cwd=None):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def build_native(log=print):
    """Compile experiments/native/vides_native.c and link it against the
    objects `make` left in src/, so it runs exactly the compiled code the
    Python module does.  Returns the binary path.

    VIDES_NATIVE_BIN overrides this with a prebuilt binary (for example the
    emulated-GPU build from test/gpu_emulation/build_native_emu.sh)."""
    pre = os.environ.get("VIDES_NATIVE_BIN")
    if pre:
        return pre
    objs = sorted(glob.glob(os.path.join(SRC, "*.o")))
    if not objs:
        raise RuntimeError("no object files in src/ -- build ViDES first "
                           "(./build.sh or ./build.sh --gpu)")
    gpu = module_has_gpu() and os.path.exists(os.path.join(SRC, "vides_gpu.o"))
    drop = {"NanoTCAD_ViDESmod.o",
            "vides_gpu_stub.o" if gpu else "vides_gpu.o"}
    objs = [o for o in objs if os.path.basename(o) not in drop]

    bindir = os.path.join(RESULTS, "bin")
    os.makedirs(bindir, exist_ok=True)
    exe = os.path.join(bindir, "vides_native_gpu" if gpu else "vides_native_cpu")
    newest = max(os.path.getmtime(p) for p in objs + [NATIVE_SRC])
    if os.path.exists(exe) and os.path.getmtime(exe) >= newest:
        return exe

    lib = os.path.join(bindir, "libvides_native.a")
    if os.path.exists(lib):
        os.remove(lib)
    rc, out = _run(["ar", "rcs", lib] + objs)
    if rc:
        raise RuntimeError("ar failed:\n" + out)
    hobj = os.path.join(bindir, "vides_native.o")
    rc, out = _run(["gcc", "-O2", "-std=gnu99", "-w", "-I" + SRC, "-c",
                    NATIVE_SRC, "-o", hobj])
    if rc:
        raise RuntimeError("compiling the native harness failed:\n" + out)

    cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    cudalib = os.path.join(cuda, "lib64")
    gpulibs = (["-L" + cudalib, "-Wl,-rpath," + cudalib, "-lcublas", "-lcudart",
                "-lstdc++"] if gpu else [])
    base = ["gcc", "-o", exe, hobj, "-Wl,--start-group", lib]
    tails = [lp + ["-Wl,--end-group"] + gf
             for lp in (["-llapack", "-lblas"], ["-llapack"], ["-lopenblas"], [])
             for gf in (["-lgfortran"], [])]
    errs = []
    for tail in tails:
        rc, out = _run(base + tail + gpulibs + ["-lm", "-lpthread"])
        if rc == 0:
            log("  built native harness (%s backend): %s"
                % ("CUDA" if gpu else "CPU-only", os.path.relpath(exe, ROOT)))
            return exe
        errs.append(out[-1500:])
    raise RuntimeError("linking the native harness failed:\n" + errs[-1])


def run_native(exe, args, timeout=3600):
    proc = subprocess.run([exe] + [str(a) for a in args], capture_output=True,
                          text=True, timeout=timeout)
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            d = json.loads(line)
            if "error" in d:
                raise RuntimeError("vides_native %s: %s" % (" ".join(map(str, args)), d["error"]))
            return d
    raise RuntimeError("vides_native %s failed (rc=%d):\n%s"
                       % (" ".join(map(str, args)), proc.returncode,
                          (proc.stdout + proc.stderr)[-2000:]))


# ======================================================================
# Device worker
# ======================================================================

PROFILE_RE = re.compile(r"\[ViDES profile\].*total ([0-9.eE+-]+) s = "
                        r"self-energy ([0-9.eE+-]+) s.*NEGF solve "
                        r"([0-9.eE+-]+) s")


def run_device(spec, gpu, selfh=None, repeat=1, timeout=7200):
    """Run one device end to end (charge_T + current) in a fresh process
    with the backend fixed by VIDES_GPU.  spec: see worker.py."""
    import tempfile
    env = dict(os.environ)
    env["VIDES_GPU"] = "1" if gpu else "0"
    env["VIDES_PROFILE"] = "1"
    if selfh:
        env["VIDES_SELFH"] = selfh
    else:
        env.pop("VIDES_SELFH", None)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [SRC, DEMO, env.get("PYTHONPATH", "")] if p)
    fd, out = tempfile.mkstemp(suffix=".json", prefix="vides_exp_")
    os.close(fd)
    try:
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "worker.py"),
             json.dumps(spec), str(repeat), out],
            env=env, capture_output=True, text=True, timeout=timeout, cwd=SRC)
        if proc.returncode != 0:
            raise RuntimeError("device run %s (%s) failed:\n%s"
                               % (spec, "GPU" if gpu else "CPU",
                                  (proc.stdout + proc.stderr)[-2500:]))
        with open(out) as f:
            info = json.load(f)
    finally:
        os.remove(out)
    info["backend"] = ""
    info["profile"] = []
    info["selfh_line"] = ""
    for line in proc.stdout.splitlines():
        if "NEGF backend" in line and not info["backend"]:
            info["backend"] = line.strip()
        if "Contact self-energy" in line and not info["selfh_line"]:
            info["selfh_line"] = line.strip()
        m = PROFILE_RE.search(line)
        if m:
            info["profile"].append([float(g) for g in m.groups()])
    info["gpu_requested"] = bool(gpu)
    info["gpu_used"] = bool(gpu) and gpu_line_ok(info["backend"])
    return info


def gpu_line_ok(line):
    """Does the backend banner say a CUDA device is in use?"""
    l = line.lower()
    return bool(line) and not any(k in l for k in (
        "disabled", "no cuda", "cpu only", "unreadable", "built without"))


# ======================================================================
# Estimates and ETA
# ======================================================================

class CostModel:
    """Seconds = overhead + scale * units, per cost class.  Scales start
    from a short calibration run and are refined after every finished
    step (running geometric mean of the measured seconds-per-unit), so the
    estimates and the ETA track the machine actually in use."""

    OVERHEAD = {"native": 0.05, "native_gpu": 0.8,
                "device": 1.5, "device_gpu": 3.0}

    def __init__(self):
        self.scale = {"_default": 2e-9}
        self.n = {}

    @staticmethod
    def _oh(cls, gpu):
        return CostModel.OVERHEAD["%s%s" % ("device" if cls.startswith("dev") else "native",
                                             "_gpu" if gpu else "")]

    def _scale(self, cls):
        # most specific first: dev_gnr_gpu -> dev_gpu -> dev -> _default
        parts = cls.split("_")
        cands = [cls]
        if len(parts) > 2:
            cands.append(parts[0] + "_" + parts[-1])
        cands += [parts[0], "_default"]
        for c in cands:
            if c in self.scale:
                return self.scale[c]
        return self.scale["_default"]

    def seconds(self, cls, units, gpu=False):
        return self._oh(cls, gpu) + self._scale(cls) * units

    def update(self, cls, units, gpu, actual, also=()):
        if units <= 0 or actual is None:
            return
        oh = self._oh(cls, gpu)
        s = max(actual - oh, 0.3 * actual) / units
        for c in (cls,) + tuple(also):
            if c in self.scale and c in self.n:
                k = min(self.n[c], 4)
                self.scale[c] = math.exp((k * math.log(self.scale[c]) + math.log(s)) / (k + 1))
                self.n[c] += 1
            else:
                self.scale[c] = s
                self.n[c] = 1


class Step:
    """One unit of work inside an experiment.

    cls/units/gpu feed the cost model; fn() does the work and returns a
    JSON-able record."""

    def __init__(self, label, cls, units, fn, gpu=False):
        self.label, self.cls, self.units, self.fn, self.gpu = label, cls, units, fn, gpu
        self.est = None
        self.est_at_start = None
        self.actual = None


def fmt_t(t):
    if t is None or t != t:
        return "  --  "
    t = float(t)
    if t <= 0:
        return "0 s"
    if t < 0.01:
        return "%.2f ms" % (1e3 * t)
    if t < 1:
        return "%4.0f ms" % (1e3 * t)
    if t < 60:
        return "%5.1f s" % t
    if t < 3600:
        return "%2dm%02ds" % (t // 60, t % 60)
    return "%dh%02dm" % (t // 3600, (t % 3600) // 60)


class Progress:
    """Prints the plan and live per-step progress with experiment and
    overall ETAs.  Works in a terminal and in a Colab/Jupyter cell (no
    cursor control: one line per finished step)."""

    def __init__(self, experiments, model, stream=sys.stdout):
        self.exps = experiments  # list of (exp, steps)
        self.model = model
        self.out = stream
        self.t0 = None

    def reestimate(self):
        for _, steps in self.exps:
            for s in steps:
                if s.actual is None:
                    s.est = self.model.seconds(s.cls, s.units, s.gpu)

    def remaining(self, steps):
        return sum(s.est for s in steps if s.actual is None)

    def total_remaining(self):
        return sum(self.remaining(st) for _, st in self.exps)

    def plan(self):
        self.reestimate()
        w = self.out.write
        w("\nPlan (estimates from a calibration run; refined as steps finish)\n")
        w("  %-4s %-48s %6s %10s\n" % ("id", "experiment", "steps", "estimate"))
        w("  " + "-" * 72 + "\n")
        for e, st in self.exps:
            w("  %-4s %-48s %6d %10s\n" % (e.ID, e.TITLE[:48], len(st), fmt_t(self.remaining(st))))
        w("  " + "-" * 72 + "\n")
        w("  %-4s %-48s %6d %10s\n\n" % ("", "total (ETA)", sum(len(s) for _, s in self.exps),
                                        fmt_t(self.total_remaining())))
        self.out.flush()

    def start(self):
        self.t0 = time.time()

    def exp_start(self, e, steps):
        self.reestimate()
        self.out.write("== %s  %s  [%d steps, est. %s | all remaining: %s]\n"
                       % (e.ID, e.TITLE, len(steps), fmt_t(self.remaining(steps)),
                          fmt_t(self.total_remaining())))
        self.out.flush()

    def step_done(self, e, steps, i, ok=True):
        s = steps[i]
        parts = s.cls.split("_")
        also = (parts[0] + "_" + parts[-1],) if len(parts) > 2 else ()
        self.model.update(s.cls, s.units, s.gpu, s.actual, also)
        self.reestimate()
        el = time.time() - self.t0
        rem_all = self.total_remaining()
        frac = el / (el + rem_all) if el + rem_all > 0 else 1
        bar = "#" * int(20 * frac) + "." * (20 - int(20 * frac))
        self.out.write("   [%2d/%2d] %-40s %8s (est %8s)%s | %s ETA %8s | all [%s] %3.0f%% ETA %8s\n"
                       % (i + 1, len(steps), s.label[:40], fmt_t(s.actual), fmt_t(s.est_at_start),
                          "" if ok else " FAILED", e.ID, fmt_t(self.remaining(steps)),
                          bar, 100 * frac, fmt_t(rem_all)))
        self.out.flush()


# ======================================================================
# Numbers
# ======================================================================

def rel_to_peak(a, b):
    """max|a-b| / max(|a|,|b|) over two equal-length sequences."""
    if len(a) != len(b):
        return float("inf")
    if not a:
        return 0.0
    d = max(abs(x - y) for x, y in zip(a, b))
    p = max(max(abs(x) for x in a), max(abs(y) for y in b))
    return d / p if p > 0 else d


def finite(v):
    return v is not None and v == v and v not in (float("inf"), float("-inf"))


def fmt_e(v):
    return "%.1e" % v if finite(v) else "n/a"


# ======================================================================
# Plot style
# ======================================================================

CPU_C = "#2a78d6"
GPU_C = "#eb6834"
ALT_C = ["#2a78d6", "#eb6834", "#3a9d5d", "#8e5cc7", "#c9a227", "#d64b7a"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = ("#0b0b0b", "#52514e", "#898781",
                                         "#e1e0d9", "#c3c2b7", "#fcfcfb")


def plt():
    import matplotlib
    if not _in_notebook():
        matplotlib.use("Agg")
    import matplotlib.pyplot as p
    p.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "legend.frameon": False,
        "legend.fontsize": 8, "figure.dpi": 110,
    })
    return p


def _in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ("IPKernelApp" in getattr(ip, "config", {})
                                   or "google.colab" in sys.modules)
    except Exception:
        return False


def in_notebook():
    return _in_notebook()


def save_fig(fig, name):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return path


def positive(v, floor=1e-17):
    """Clamp for log axes: zeros (exact agreement) become `floor`."""
    return [max(x, floor) if finite(x) else float("nan") for x in v]


def save_json(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=1)


def load_json(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def which(x):
    return shutil.which(x)


# ======================================================================
# Work units for the cost model (rough operation counts, so classes that
# share a scale stay comparable before they are refined)
# ======================================================================

def u_rgf(n, Nc, NB):
    return 10.0 * NB * Nc * n ** 3


def u_decim(n, NB):
    return 150.0 * NB * (4 * n) ** 3


def u_selfh(n, NE):
    return 160.0 * NE * (4 * n) ** 3


def device_dims(spec):
    """(n, Nc, NE) a device spec will have, without building it."""
    kind, (a, b) = spec["kind"], spec["size"]
    lo, hi, dE = spec["grid"]
    NE = int(round((hi - lo) / dE)) + 1
    if kind in ("cnt", "cntmode", "gnr"):
        Nc = int(4 * ((int(float(b) / 0.144) - 1) // 3) + 2)
        n = int(a) if kind != "cntmode" else int(spec.get("nmodes", 4))
    else:
        n, Nc = int(a), int(b)
    return n, Nc, NE


def u_device(spec, gpu=False, selfh=None, repeat=1):
    n, Nc, NE = device_dims(spec)
    kind = spec["kind"]
    if kind == "gnr":
        sig = 2 * 150.0 * (4 * n) ** 3
    elif kind == "hamiltonian":
        dec = selfh == "dec" or (selfh is None and gpu)
        sig = 2 * (150.0 * (4 * n) ** 3 if dec else 300.0 * (2 * n) ** 3)
    else:
        sig = 50.0 * n ** 3
    return repeat * NE * (10.0 * Nc * n ** 3 + sig)


def device_step(label, spec, gpu, selfh=None, repeat=1, extra=None):
    """Step running one device on one backend; the record carries `tag`
    fields from `extra` so analyses can find it."""
    def fn():
        r = run_device(spec, gpu, selfh=selfh, repeat=repeat)
        r.update(extra or {})
        r["selfh"] = selfh
        return r
    return Step(label, "dev_%s_%s" % (spec["kind"], "gpu" if gpu else "cpu"),
                u_device(spec, gpu, selfh, repeat), fn, gpu)


def steady(times):
    """Best time after the first call (which carries GPU start-up)."""
    return min(times[1:] or times)
