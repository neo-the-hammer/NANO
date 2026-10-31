"""Power sampling for energy-to-solution measurements.

GPU: nvidia-smi samples the board power every `period` ms in a
background process (power.draw, W).  CPU: Linux RAPL counters
(/sys/class/powercap/intel-rapl:N/energy_uj, the package domains), read
at start and stop; most cloud VMs and containers hide them, and then the
CPU side is reported as unavailable instead of guessed.

    s = PowerSampler()
    s.start(); ...work...; r = s.stop()
    r = {"seconds", "gpu_w_mean", "gpu_j", "gpu_samples", "cpu_j", "cpu_w_mean"}

idle(seconds) measures the same quantities with nothing running, so a
result can be given both as total and as above-idle power.
"""

import glob
import os
import subprocess
import threading
import time


def _rapl_domains():
    out = []
    for d in sorted(glob.glob("/sys/class/powercap/intel-rapl:[0-9]*")):
        if os.path.basename(d).count(":") != 1:      # package domains only, not sub-domains
            continue
        f = os.path.join(d, "energy_uj")
        mx = os.path.join(d, "max_energy_range_uj")
        try:
            with open(f) as h:
                int(h.read())
            rng = int(open(mx).read()) if os.path.exists(mx) else 2 ** 32
            out.append((f, rng))
        except (OSError, ValueError):
            continue
    return out


def _rapl_read(domains):
    v = []
    for f, _ in domains:
        try:
            with open(f) as h:
                v.append(int(h.read()))
        except (OSError, ValueError):
            v.append(None)
    return v


def gpu_power_available():
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return p.returncode == 0 and any(_num(x) is not None for x in p.stdout.splitlines())
    except Exception:
        return False


def _num(s):
    try:
        return float(s.strip().split()[0])
    except (ValueError, IndexError):
        return None


class PowerSampler:
    def __init__(self, period_ms=100):
        self.period = period_ms
        self.rapl = _rapl_domains()
        self.gpu_ok = gpu_power_available()
        self._proc = None
        self._samples = []
        self._thread = None

    def _reader(self):
        for line in self._proc.stdout:
            vals = [_num(x) for x in line.strip().split(",")]
            vals = [x for x in vals if x is not None]
            if vals:
                self._samples.append((time.time(), sum(vals)))   # every GPU of the node

    def start(self):
        self._samples = []
        self._t0 = time.time()
        self._e0 = _rapl_read(self.rapl)
        if self.gpu_ok:
            self._proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
                 "-lms", str(self.period)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        t1 = time.time()
        e1 = _rapl_read(self.rapl)
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            if self._thread:
                self._thread.join(timeout=2)
        dt = max(t1 - self._t0, 1e-9)
        r = {"seconds": dt, "gpu_samples": len(self._samples), "gpu_w_mean": None, "gpu_j": None,
             "cpu_j": None, "cpu_w_mean": None}
        if self._samples:
            w = [s for _, s in self._samples]
            r["gpu_w_mean"] = sum(w) / len(w)
            r["gpu_j"] = r["gpu_w_mean"] * dt
        if self.rapl and None not in self._e0 and None not in e1:
            j = 0.0
            for (f, rng), a, b in zip(self.rapl, self._e0, e1):
                d = b - a
                if d < 0:
                    d += rng                     # counter wrapped
                j += d * 1e-6
            r["cpu_j"] = j
            r["cpu_w_mean"] = j / dt
        return r


def idle(seconds=3.0):
    s = PowerSampler().start()
    time.sleep(seconds)
    return s.stop()
