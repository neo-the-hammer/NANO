# CPU vs GPU experiments

One command runs every experiment, shows the results, and compares the two
backends visually:

```bash
./build.sh --gpu                              # once: module + CUDA backend
python3 experiments/run_all.py --quick        # a few minutes
python3 experiments/run_all.py                # full sizes
```

In Colab or Jupyter, run it inside the kernel so the figures appear inline:

```python
%cd /content/NanoTCAD_ViDES
%run experiments/run_all.py --quick
```

## What you see

1. **Plan.** After a ~5 s calibration on your machine, a table of the
   experiments with the number of steps and the estimated time of each,
   and the total ETA.
2. **Progress.** One line per finished step: its time against its
   estimate, the time left in the current experiment, and the overall
   ETA with a progress bar. Estimates are refined after each step from
   the times actually measured, so the ETA tightens as the run goes on.
3. **Per experiment.** A verdict (PASS / WARN / FAIL for correctness and
   accuracy, INFO for speed), a table, and a figure comparing CPU and GPU.
4. **Summary.** One line per experiment: verdict, time taken vs
   estimated, headline number.

Everything is saved to `experiments/results/`: one JSON per experiment
with all raw data, one PNG per figure, and `report.html` with the summary,
every table and every figure in one page.

## The experiments

| id | kind | what it compares |
|---|---|---|
| C1 | correctness | The NEGF solver (recursive Green's function) alone: CPU reference vs batched cuBLAS, all three variants (std, Lake, mode space), two block sizes, plus a near-singular case (eta = 1e-9). Compares source LDOS, drain LDOS and transmission. |
| C2 | correctness | The contact self-energy alone (Lopez-Sancho decimation): CPU `Gzerozero` vs batched GPU decimation, on the GNR lead and both Hamiltonian-path leads. Also scores each against the lead's own fixed-point equation. |
| C3 | correctness | Whole devices through the Python API, CPU process vs GPU process: CNT, CNT mode space, GNR, generic Hamiltonian (both on decimation, and each with its default self-energy). Transmission, charge per site, current. |
| A1 | accuracy | The two self-energy methods of the Hamiltonian path (eigen `selfH_new`, decimation `selfH_dec`) and the GPU decimation, scored by their residual in the lead equation, as eta goes from 1e-3 to 1e-7. |
| A2 | accuracy | Accuracy vs speed on one Hamiltonian device: CPU+eigen (reference), CPU+decimation, GPU+eigen, GPU+decimation. |
| S1 | speed | The NEGF solver: time and speedup vs block size n, number of blocks Nc and energies per batch NB. |
| S2 | speed | The self-energy: time per energy of eigen (CPU), decimation (CPU) and decimation (GPU), vs lead size and batch size. |
| S3 | speed | Whole devices, CPU vs GPU, with each solve split into self-energy, NEGF solve and the rest. |

Each module in `exps/` documents its cases, metric and pass threshold in
its docstring (also shown in the report).

**How differences are measured.** `max|cpu - gpu| / max(|cpu|, |gpu|)`
over the whole array (relative to the peak). Element-wise relative
differences are shown too, but are dominated by values that are tiny to
begin with (transmission inside a gap), so they do not decide verdicts.

**Why decimation is judged by residuals.** Decimation stops when its
coupling matrices underflow to exactly zero, which the CPU and the GPU can
reach one iteration apart. The two then agree only to the accuracy of the
method itself, so C2 checks that both results solve the lead equation
equally well, not that they agree bit for bit.

## Options

```
--quick            small sizes
--only C1 S1 ...   run only these
--skip S3 ...      skip these
--replot           re-analyse and re-plot saved results, no runs
--no-plots         tables only
--list             list the experiments
```

## How it works

- `native/vides_native.c` calls the internal processes (batched solver,
  batched decimation, both self-energy methods) directly, CPU and GPU in
  one process with identical deterministic inputs. `run_all.py` compiles
  it and links it against the objects `make` left in `src/`, so it runs
  exactly the code the Python module runs.
- `worker.py` runs one device through the Python API; each backend gets
  its own process, because the backend is chosen once per process
  (`VIDES_GPU`). `VIDES_PROFILE=1` supplies the time breakdown.
- Without a GPU everything still runs; GPU columns are empty and the
  correctness experiments report NO GPU.
- To exercise the suite on a machine with no GPU at all, build the
  harness against the emulated GPU in `test/gpu_emulation/` (its timings
  mean nothing, its numbers do):

  ```bash
  sh test/gpu_emulation/build_native_emu.sh /tmp/vides_native_emu
  VIDES_NATIVE_BIN=/tmp/vides_native_emu python3 experiments/run_all.py --quick --only C1 C2 A1 S1 S2
  ```
