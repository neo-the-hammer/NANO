#!/usr/bin/env python3
"""Reproducibility manifest: the machine, the software and the code version.

A timing is only meaningful next to the hardware and libraries that
produced it, so every report of the publication suite starts with this
manifest, and it is saved as environment.json next to the results:

  CPU model, physical cores and logical threads, memory
  GPU model(s), driver, CUDA runtime and compiler, compute capability
  the BLAS / LAPACK libraries the Python module links (or "bundled": the
  reference LAPACK compiled into the module)
  Python, NumPy, SciPy and matplotlib versions
  the git commit of the code and whether the tree has local changes

    python3 experiments/paper/environment.py          # print it
    python3 experiments/paper/environment.py --json   # as JSON
"""

import json
import os
import platform
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "src")


def _cmd(cmd):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20)
        return p.stdout.strip() if p.returncode == 0 else ""
    except Exception:
        return ""


def cpu():
    model, sockets, cores_per, threads = "", 1, 0, os.cpu_count() or 1
    txt = _cmd("lscpu")
    for line in txt.splitlines():
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if k == "Model name":
            model = v
        elif k == "Socket(s)":
            sockets = int(v) if v.isdigit() else sockets
        elif k == "Core(s) per socket":
            cores_per = int(v) if v.isdigit() else cores_per
    if not model:
        try:
            with open("/proc/cpuinfo") as f:
                m = re.search(r"model name\s*:\s*(.*)", f.read())
                model = m.group(1) if m else platform.processor()
        except OSError:
            model = platform.processor() or platform.machine()
    mem = ""
    try:
        with open("/proc/meminfo") as f:
            m = re.search(r"MemTotal:\s*(\d+)", f.read())
            mem = "%.1f GB" % (int(m.group(1)) / 1048576.0) if m else ""
    except OSError:
        pass
    return {"model": model, "physical_cores": sockets * cores_per or None,
            "logical_threads": threads, "memory": mem}


def gpus():
    out = []
    txt = _cmd("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap,power.limit "
               "--format=csv,noheader")
    for line in txt.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 4:
            out.append({"name": f[0], "memory": f[1], "driver": f[2], "compute_capability": f[3],
                        "power_limit": f[4] if len(f) > 4 else ""})
    return out


def cuda():
    v = _cmd("nvcc --version")
    m = re.search(r"release ([0-9.]+)", v)
    smi = _cmd("nvidia-smi")
    r = re.search(r"CUDA Version:\s*([0-9.]+)", smi)
    return {"nvcc": m.group(1) if m else "", "driver_cuda": r.group(1) if r else ""}


def blas():
    so = os.path.join(SRC, "NanoTCAD_ViDESmod.so")
    if not os.path.exists(so):
        return "module not built"
    libs = _cmd("ldd %s" % so)
    names = sorted({m.group(1) for m in re.finditer(r"(lib(?:open)?blas[^ ]*|liblapack[^ ]*|libmkl[^ ]*|"
                                                    r"libgoto[^ ]*|libcublas[^ ]*)", libs)})
    return ", ".join(names) if names else "bundled reference BLAS/LAPACK (compiled into the module)"


def versions():
    v = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "matplotlib"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = "not installed"
    return v


def git():
    commit = _cmd("git -C %s rev-parse --short HEAD" % ROOT)
    dirty = bool(_cmd("git -C %s status --porcelain --untracked-files=no" % ROOT))
    branch = _cmd("git -C %s rev-parse --abbrev-ref HEAD" % ROOT)
    return {"commit": commit or "unknown", "branch": branch, "local_changes": dirty}


def manifest():
    return {"host": platform.node(), "os": platform.platform(), "cpu": cpu(), "gpus": gpus(),
            "cuda": cuda(), "blas": blas(), "versions": versions(), "git": git(),
            "env": {k: v for k, v in os.environ.items() if k.startswith("VIDES_")}}


def header_lines(m=None):
    m = m or manifest()
    c = m["cpu"]
    lines = ["CPU: %s, %s physical cores, %d threads, %s RAM"
             % (c["model"], c["physical_cores"] or "?", c["logical_threads"], c["memory"] or "?")]
    for g in m["gpus"]:
        lines.append("GPU: %s, %s, driver %s, compute capability %s, power limit %s"
                     % (g["name"], g["memory"], g["driver"], g["compute_capability"], g["power_limit"]))
    if not m["gpus"]:
        lines.append("GPU: none visible to nvidia-smi")
    cu = m["cuda"]
    lines.append("CUDA: nvcc %s, driver supports %s" % (cu["nvcc"] or "-", cu["driver_cuda"] or "-"))
    lines.append("BLAS/LAPACK of the module: %s" % m["blas"])
    v = m["versions"]
    lines.append("Python %s, NumPy %s, SciPy %s, matplotlib %s"
                 % (v["python"], v["numpy"], v["scipy"], v["matplotlib"]))
    gi = m["git"]
    lines.append("code: commit %s (%s)%s" % (gi["commit"], gi["branch"],
                                            ", with local changes" if gi["local_changes"] else ""))
    if m["env"]:
        lines.append("settings: " + " ".join("%s=%s" % kv for kv in sorted(m["env"].items())))
    return lines


def main():
    m = manifest()
    if "--json" in sys.argv:
        print(json.dumps(m, indent=1))
    else:
        print("\n".join(header_lines(m)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
