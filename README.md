# NanoTCAD ViDES — Python 3.13 / NumPy 2.x / GCC 14 Compatible Fork

**NanoTCAD ViDES** is a self-consistent NEGF (Non-Equilibrium Green's Function) + Poisson solver for nanoscale device simulation, developed by Prof. Gianluca Fiori and Prof. Giuseppe Iannaccone at the University of Pisa.

This fork patches the original source code to compile and run on modern toolchains:

- **Python 3.13** (was: Python 2.7)
- **NumPy 2.x** (was: NumPy 1.x with deprecated C API)
- **GCC 14** (was: older GCC with relaxed defaults)

## What Was Patched

### Python 2 → 3 C API

| Old API (Python 2) | Replacement (Python 3) | Method |
|---|---|---|
| `PyInt_AsLong()` | `PyLong_AsLong()` | Preprocessor macro |
| `PyInt_FromLong()` | `PyLong_FromLong()` | Preprocessor macro |
| `PyString_AsString()` | `PyBytes_AsString(PyUnicode_AsEncodedString(...))` | Preprocessor macro + `#if PY_MAJOR_VERSION` guards |
| `Py_InitModule()` | `PyModule_Create()` with `PyModuleDef` | Conditional compilation in `NanoTCAD_ViDESmod.c` |

### NumPy 2.x C API

| Old API | Replacement | Method |
|---|---|---|
| `PyArray_DOUBLE` | `NPY_DOUBLE` | Preprocessor macro |
| `PyArray_CDOUBLE` | `NPY_CDOUBLE` | Preprocessor macro |
| `PyArray_INT` | `NPY_INT` | Preprocessor macro |
| `PyArray_FromDimsAndData()` | `PyArray_SimpleNewFromData()` | Compatibility shim in `compat_array.h` |
| NumPy headers at `numpy/core/include/` | `numpy/_core/include/` | Updated path in `configure.py` |

### GCC 14 Fortran/C Compatibility

| Flag | Purpose |
|---|---|
| `-std=gnu89` | Allow legacy C89 constructs |
| `-Wno-implicit-function-declaration` | Suppress missing prototype warnings |
| `-Wno-int-conversion` | Allow implicit int/pointer conversions |
| `-Wno-incompatible-pointer-types` | Allow mismatched pointer assignments |
| `-fallow-argument-mismatch` | Fortran: relax strict argument checking |
| `-std=legacy` | Fortran: allow F77 syntax |

### New Files Added

- `src/vides_compat.h` — Master compatibility header
- `src/compat_array.h` — `PyArray_FromDimsAndData()` replacement for NumPy 2.x

## Building

```bash
cd src
python3 configure.py
make
make install  # installs to lib/
```

The compiled shared library `NanoTCAD_ViDESmod.so` and Python wrapper `NanoTCAD_ViDES.py` are placed in `lib/`.

## Usage

```python
import sys
sys.path.insert(0, 'lib')
from NanoTCAD_ViDES import *

# Create a (10,0) zigzag CNT, 15 nm long
CNT = nanotube(10, 15)
print(f"Bandgap: {CNT.gap():.4f} eV")  # 0.9795 eV
```

See `demo/` for tutorial scripts (Id-Vgs sweeps, Poisson solutions, Schottky contacts, etc.).

## GPU acceleration (CUDA)

The NEGF solver can optionally run its energy sweep on an NVIDIA GPU. This
is **opt-in**: without `GPU=1` the build and the numerics are exactly what
they were before.

### What is actually accelerated

All four device paths reach the same place — a sweep over energy whose body
is a Recursive Green's Function recursion (`rgfblock.c`) over `Nc` blocks of
small `n x n` complex matrices:

| Entry point | Class | Recursion |
|---|---|---|
| `CNT_charge_T` | `nanotube.charge_T()` | `LDOS` (standard) |
| `CNTmode_charge_T` | `nanotube.mode_charge_T()` | `LDOSMODE` (mode space) |
| `GNR_charge_T` | `nanoribbon.charge_T()` | `LDOS` (standard) |
| `H_charge_T` | `Hamiltonian`, `Zincblend` | `LDOS_Lake` |

A single `n x n` inversion — `n` is typically 20–200 — is far too small to
fill a GPU, and moving one across PCIe per energy point would be slower than
staying on the CPU. The energy points, however, are completely independent.
So the GPU backend keeps the recursion sequential in the block index and
batches over **energy** instead: each cuBLAS call (`cublasZgetrfBatched`,
`cublasZgetriBatched`, `cublasZgemmStridedBatched`) handles hundreds of
matrices at once. The batch size is chosen at run time from free device
memory.

The energy-independent Hamiltonian blocks are uploaded once per batch and
shared across the whole batch with a zero stride, so only the self-energies
scale with the batch.

### Building

```bash
./build.sh --gpu            # or: cd src && python3 configure.py && make GPU=1
```

`build.sh` is a POSIX replacement for `install.sh` (which is a tcsh script,
and so does not run on Colab). If `nvcc` rejects the default architecture
list, override it:

```bash
CUDA_ARCH="-arch=sm_75" ./build.sh --gpu
```

The default list targets Pascal through Ampere with a PTX fallback, and uses
only APIs available in CUDA 11.

### On Google Colab

```python
!apt-get -qq install -y gfortran liblapack-dev libblas-dev
!cd /content/NanoTCAD_ViDES && ./build.sh --gpu
import sys; sys.path.insert(0, '/content/NanoTCAD_ViDES/src')
from NanoTCAD_ViDES import *
```

A word on what to expect: Colab's default runtime is a **T4**, which runs
FP64 at 1/32 of its FP32 rate (~0.25 TFLOPS) — comparable to a good multicore
CPU. The whole solver is FP64, matching the CPU path, so the gain on a T4
comes from replacing thousands of launch-latency-bound tiny solves with a few
throughput-bound batched ones, not from raw arithmetic. Expect a solid but
not dramatic speedup there, and considerably more on a V100 or A100 runtime,
whose FP64 rate is 1/2. The speedup also grows with `n`, `Nc` and the number
of energy points; for a very small device the CPU may still win.

### Controlling it at run time

| Variable | Effect |
|---|---|
| `VIDES_GPU=0` | Force the CPU path even on a GPU build |
| `VIDES_GPU_BATCH=N` | Cap the number of energies per batch |

Each NEGF call prints which backend it selected and the batch size. If the
GPU fails mid-run (out of memory, a singular block in the batched LU), the
driver prints a warning and redoes that batch on the CPU rather than losing
the bias point.

### Verifying it

Correctness is not assumed. The CPU and GPU paths implement the same
recursion, so they must agree to round-off:

```bash
cd src && PYTHONPATH=. python3 ../test/test_gpu_vs_cpu.py
```

This runs each device path twice — once with `VIDES_GPU=0`, once with it
enabled — and reports the largest absolute and relative deviation in charge,
transmission and the energy grid. They will not match bit for bit, since a
batched cuBLAS gemm sums in a different order than the reference `zgemm`;
the default tolerance is `1e-6` relative and the actual deviation is always
printed so you can judge it yourself.

There is also a host-side type check that needs no GPU and no CUDA toolkit:

```bash
sh test/gpu_syntax_check/check_syntax.sh
```

And, more usefully, the GPU driver can be **executed and checked without a
GPU at all**:

```bash
sh test/gpu_emulation/run_test.sh
```

This emulates the CUDA runtime and the cuBLAS batched calls on the CPU
(device memory is host memory; kernel launches are replayed serially over
the grid), compiles the real `vides_gpu.cu` against it, and runs the actual
`vides_rgf_batch_gpu()` against the stock `LDOS` / `LDOS_Lake` / `LDOSMODE`
path on identical input. Only the launch configuration is rewritten — the
algorithm, the indexing and the cuBLAS argument order under test are the
shipped ones. It covers all three recursion variants, degenerate shapes
(`Nc=2`, `NB=1`), device-memory accounting, and the batch-size chooser.

What it cannot cover: `nvcc` codegen, real CUDA semantics (races,
coalescing, launch limits), and the `__syncthreads()` reduction in
`k_trace_prod`, which is launched with one thread per block under emulation
— the reduction tree is checked separately instead.

### Status

**Verified on a real GPU** (Google Colab, Tesla T4, CUDA, `sm_75`): all four
device paths run on the GPU and agree with the CPU path to round-off.

| Case | Charge, relative to peak | Transmission, relative to peak |
|---|---|---|
| CNT (`LDOS`) | 1.5e-10 | 1.8e-10 |
| CNT mode space (`LDOSMODE`) | 1.2e-15 | 1.6e-15 |
| GNR (`LDOS`) | 3.8e-09 | 1.5e-15 |
| Hamiltonian / nanowire (`LDOS_Lake`) | 2.7e-11 | 2.0e-15 |

One caveat applies to both backends, not just the GPU. If an energy point
lands exactly on a near-singular resonance — for example a graphene
ribbon's zigzag edge states at the band centre, with a small `eta` — the
inversion there is ill-conditioned and the result at that one energy is
accurate to only a few digits. The CPU and GPU then disagree there (on the
T4, by ~7e-7 in integrated charge), because cuBLAS and LAPACK round
differently and the conditioning amplifies it. Neither value is the
"right" one. Shifting the grid by half a step, or using a larger `eta`,
removes it. `test_gpu_vs_cpu.py` keeps such a case as a diagnostic
(`hamiltonian_ongrid`).

Speed is measured with `test/benchmark_gpu.py` (below).

### Benchmarking

```bash
cd src && PYTHONPATH=. python3 ../test/benchmark_gpu.py --quick   # a few minutes
cd src && PYTHONPATH=. python3 ../test/benchmark_gpu.py           # full sweep
```

For each path it times the same charge/transmission solve on the CPU and
the GPU across growing devices, and prints block size `n`, block count
`Nc`, energy points `NE`, the times, and the speedup. The first GPU call
in a process includes one-off CUDA start-up and is reported separately
("GPU first"); the speedup uses the steady-state time, which is what each
further solve in a self-consistent loop costs.

The results are also saved to `benchmark_results.json`. To plot them:

```bash
cd src && python3 ../test/plot_benchmark.py benchmark_results.json   # writes benchmark.png
```

One column per device path: time per solve on a log scale (CPU, GPU
steady state, GPU first call) above the speedup, with the 1x break-even
line marked. The same numbers are printed as a table.

### Files

| File | Role |
|---|---|
| `src/vides_rgf_batch.h` | Batched NEGF API shared by both backends |
| `src/vides_rgf_batch.c` | CPU reference (calls the stock `LDOS`) and dispatcher |
| `src/vides_gpu.cu` | CUDA backend |
| `src/vides_gpu_stub.c` | No-op stubs for CPU-only builds |
| `test/test_gpu_vs_cpu.py` | GPU-vs-CPU numerical comparison |
| `test/gpu_syntax_check/` | Type check for `vides_gpu.cu` without CUDA |
| `test/benchmark_gpu.py` | CPU vs GPU timing across device sizes |
| `test/plot_benchmark.py` | Plots the benchmark results |
| `test/gpu_emulation/` | Runs `vides_gpu.cu` on the CPU and checks it against the stock path |


## Credits

- **Original NanoTCAD ViDES**: Gianluca Fiori & Giuseppe Iannaccone, University of Pisa
  - Website: http://vides.nanotcad.com/vides/
  - Paper: Fiori & Iannaccone, *J. Comput. Electron.* (2023)
- **Python 3.13 / NumPy 2.x / GCC 14 patches**: Manish Jagdish Thatte

## License

Original NanoTCAD ViDES license applies. See `license.txt`.
