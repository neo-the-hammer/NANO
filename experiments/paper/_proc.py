"""Run one of this folder's worker scripts in a fresh process.

The backend is chosen once per process (VIDES_GPU), so every (worker,
backend) pair gets its own process, as in common.run_device().  The
worker writes a JSON file; its stdout carries the library's banners
("NEGF backend: ...", profile lines), which tell whether the GPU was
really used.
"""

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
import common as C  # noqa: E402


def run_worker(script, payload, gpu, env=None, timeout=7200):
    e = C.mode_env(dict(os.environ), env)
    e["VIDES_GPU"] = "1" if gpu else "0"
    e["VIDES_PROFILE"] = "1"
    e.setdefault("MPLBACKEND", "Agg")
    e["PYTHONPATH"] = os.pathsep.join(p for p in [C.SRC, C.DEMO, C.ROOT, e.get("PYTHONPATH", "")] if p)
    for k, v in (env or {}).items():
        e[k] = v
    fd, out = tempfile.mkstemp(suffix=".json", prefix="vides_paper_")
    os.close(fd)
    try:
        p = subprocess.run([sys.executable, os.path.join(HERE, script), json.dumps(payload), out],
                           env=e, capture_output=True, text=True, timeout=timeout, cwd=C.SRC)
        if p.returncode != 0 or os.path.getsize(out) == 0:
            raise RuntimeError("%s (%s) failed (rc=%d):\n%s" % (script, "GPU" if gpu else "CPU", p.returncode,
                                                               (p.stdout + p.stderr)[-2500:]))
        with open(out) as f:
            info = json.load(f)
    finally:
        os.remove(out)
    C.parse_banners(info, p.stdout, gpu)
    return info
