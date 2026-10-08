/* Run vides_gpu.cu's driver against the stock CPU path on identical input.
 *
 * Both sides are handed the same block-tridiagonal problem and the same
 * self-energies.  The CPU side goes through vides_rgf_batch_cpu(), which
 * calls the untouched LDOS() / LDOS_Lake() / LDOSMODE(); the GPU side runs
 * the real vides_rgf_batch_gpu() over the emulated CUDA runtime.  They use
 * the same underlying gemm and LU, so any disagreement beyond round-off is
 * an algorithm or indexing bug in the GPU driver -- which is exactly what
 * this is for.
 *
 * Not part of the shipped library.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "cuda_runtime.h"

extern "C" {
#include "vides_rgf_batch.h"
#include "cmatrix.h"
#include "cfree_cmatrix.h"
vides_complex **cmatinv(vides_complex **A, int N);
vides_complex **selfGNR(double E, double *Em1, int N, double thop, double eta);
void selfGNR_cell(double *Em1, int N, double thop,
                  vides_complex **W0, vides_complex **BETA, vides_complex **BETADAGA);
void selfH_dec_cell(vides_complex ***diag, vides_complex ***updiag, vides_complex ***lowdiag,
                    int N, int Nc, int lead,
                    vides_complex **W0, vides_complex **BETA, vides_complex **BETADAGA);
}

/* Deterministic PRNG so a failure is reproducible. */
static unsigned long long rng_state = 88172645463325252ULL;
static double urand(void)
{
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (double)((rng_state >> 11) & 0xFFFFFFFFULL) / (double)0xFFFFFFFFULL;
}
static double srand2(void) { return 2.0*urand() - 1.0; }

/* cmatrix()-compatible allocation: one contiguous n*n block plus row
   pointers, exactly what the real cmatrix() hands back. */
static vides_complex **newblk(int n)
{
  vides_complex *data = (vides_complex *)calloc((size_t)n*n, sizeof(vides_complex));
  vides_complex **rows = (vides_complex **)malloc((size_t)n*sizeof(vides_complex *));
  for (int i = 0; i < n; i++) rows[i] = data + (size_t)i*n;
  return rows;
}
static void freeblk(vides_complex **m) { free(m[0]); free(m); }

struct Problem {
  int n, Nc, NB;
  vides_complex ***diag, ***up, ***low, ***ss, ***sd;
  double *E;
};

/* A nearest-neighbour chain with a little disorder: Hermitian blocks, a
   hopping that couples them, and contact self-energies with a genuine
   anti-Hermitian part so gamma is non-zero. */
static void build(Problem &p, int n, int Nc, int NB, unsigned long long seed)
{
  rng_state = seed;
  p.n = n; p.Nc = Nc; p.NB = NB;
  p.diag = (vides_complex ***)malloc(Nc*sizeof(void *));
  p.up   = (vides_complex ***)malloc(Nc*sizeof(void *));
  p.low  = (vides_complex ***)malloc(Nc*sizeof(void *));
  p.ss   = (vides_complex ***)malloc(NB*sizeof(void *));
  p.sd   = (vides_complex ***)malloc(NB*sizeof(void *));
  p.E    = (double *)malloc(NB*sizeof(double));

  for (int b = 0; b < Nc; b++) {
    p.diag[b] = newblk(n); p.up[b] = newblk(n); p.low[b] = newblk(n);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        if (i == j) { p.diag[b][i][j].r = 0.3*srand2(); p.diag[b][i][j].i = 0.0; }
        else if (j == i+1 || i == j+1) {
          double t = -2.7 + 0.1*srand2();
          p.diag[b][i][j].r = t; p.diag[b][j][i].r = t;
        }
      }
    /* Couple neighbouring slices; last up / first low stay zero, as the
       entry points set them. */
    /* General complex couplings.  Multiples of the identity commute with
       everything and would hide a swapped matrix-product order, so fill
       the whole block.  low[b] = up[b-1]^dagger keeps H Hermitian. */
    if (b < Nc-1)
      for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
          p.up[b][i][j].r = (i == j ? -2.7 : 0.0) + 0.4*srand2();
          p.up[b][i][j].i = 0.3*srand2();
        }
  }
  for (int b = 1; b < Nc; b++)
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        p.low[b][i][j].r =  p.up[b-1][j][i].r;
        p.low[b][i][j].i = -p.up[b-1][j][i].i;
      }
  {
  }

  for (int e = 0; e < NB; e++) {
    p.E[e] = -1.5 + 3.0*(double)e/(double)(NB > 1 ? NB-1 : 1) + 0.017;
    p.ss[e] = newblk(n); p.sd[e] = newblk(n);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        p.ss[e][i][j].r = 0.20*srand2(); p.ss[e][i][j].i = -0.15*fabs(srand2()) - 0.05;
        p.sd[e][i][j].r = 0.20*srand2(); p.sd[e][i][j].i = -0.15*fabs(srand2()) - 0.05;
      }
  }
}

static void destroy(Problem &p)
{
  for (int b = 0; b < p.Nc; b++) { freeblk(p.diag[b]); freeblk(p.up[b]); freeblk(p.low[b]); }
  for (int e = 0; e < p.NB; e++) { freeblk(p.ss[e]); freeblk(p.sd[e]); }
  free(p.diag); free(p.up); free(p.low); free(p.ss); free(p.sd); free(p.E);
}

static int compare(const char *what, const double *a, const double *b, size_t N,
                   double tol)
{
  double maxabs = 0.0, maxrel = 0.0, scale = 0.0;
  size_t worst = 0;
  for (size_t i = 0; i < N; i++) {
    double d = fabs(a[i]-b[i]);
    double s = fabs(a[i]) > fabs(b[i]) ? fabs(a[i]) : fabs(b[i]);
    if (s > scale) scale = s;
    if (d > maxabs) { maxabs = d; worst = i; }
    if (s > 1e-12 && d/s > maxrel) maxrel = d/s;
  }
  int ok = (maxrel <= tol) || (maxabs <= 1e-13);
  printf("    %-4s  n=%-7zu  max|d|=%-11.4g  max rel=%-11.4g  scale=%-10.4g  %s\n",
         what, N, maxabs, maxrel, scale, ok ? "ok" : "MISMATCH");
  if (!ok)
    printf("          worst at index %zu: cpu=%.17g  gpu=%.17g\n",
           worst, a[worst], b[worst]);
  return ok;
}

static int run_case(const char *label, int variant, int n, int Nc, int NB,
                    int Nreal, const int *order, double tol)
{
  Problem p;
  build(p, n, Nc, NB, 0x9E3779B97F4A7C15ULL ^ (unsigned long long)variant);

  vides_rgf_desc d;
  memset(&d, 0, sizeof d);
  d.n = n; d.Nc = Nc; d.NB = NB;
  d.variant = variant; d.flagtrans = 1; d.eta = 1e-3;
  d.Nreal = Nreal; d.order = order;

  int nout = (variant == VIDES_RGF_MODE) ? Nreal : n;
  size_t NA = (size_t)NB*Nc*nout;

  double *A1c = (double *)calloc(NA, sizeof(double));
  double *A2c = (double *)calloc(NA, sizeof(double));
  double *Tc  = (double *)calloc(NB, sizeof(double));
  double *A1g = (double *)calloc(NA, sizeof(double));
  double *A2g = (double *)calloc(NA, sizeof(double));
  double *Tg  = (double *)calloc(NB, sizeof(double));

  printf("  %s  (n=%d Nc=%d NB=%d nout=%d)\n", label, n, Nc, NB, nout);

  int rc_cpu = vides_rgf_batch_cpu(&d, p.E, p.diag, p.up, p.low, p.ss, p.sd,
                                   A1c, A2c, Tc);
  size_t before = vides_emu_live_bytes();
  int rc_gpu = vides_rgf_batch_gpu(&d, p.E, p.diag, p.up, p.low, p.ss, p.sd,
                                   A1g, A2g, Tg);
  size_t after = vides_emu_live_bytes();

  int ok = 1;
  if (rc_cpu != 0) { printf("    CPU path returned %d\n", rc_cpu); ok = 0; }
  if (rc_gpu != 0) { printf("    GPU path returned %d\n", rc_gpu); ok = 0; }

  if (ok) {
    ok &= compare("A1", A1c, A1g, NA, tol);
    ok &= compare("A2", A2c, A2g, NA, tol);
    ok &= compare("T",  Tc,  Tg,  (size_t)NB, tol);
  }
  if (after != before) {
    printf("    LEAK: %zu device bytes still held (%d allocations)\n",
           after - before, vides_emu_live_allocs());
    ok = 0;
  }

  free(A1c); free(A2c); free(Tc); free(A1g); free(A2g); free(Tg);
  destroy(p);
  return ok;
}

/* ---- batched decimation (contact self-energies) ------------------- */

static double blk_diff(vides_complex **a, vides_complex **b, int n, double *scale)
{
  double d = 0, m = 0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      d = fmax(d, hypot(a[i][j].r - b[i][j].r, a[i][j].i - b[i][j].i));
      m = fmax(m, hypot(a[i][j].r, a[i][j].i));
    }
  *scale = m;
  return d;
}

static void free_out(vides_complex ***o, int NB, int n)
{ for (int b = 0; b < NB; b++) cfree_cmatrix(o[b], 0, n - 1, 0, n - 1); }

/* Residual of the lead's fixed-point equation for one self-energy
   sigma (N x N block at (3,3) of the cell), relative to |sigma|:
     sigma = BETA_30 [ (W0 + (E + i eta) I - sigma@(3,3))^-1 ]_00 BETADAGA_03
   The exact answer satisfies it whatever method produced it, so it
   arbitrates between GPU and CPU when they differ. */
static double lead_residual(vides_complex **sig, double E, double eta, int N,
                            vides_complex **W0, vides_complex **B, vides_complex **BD)
{
  const int M = 4 * N;
  vides_complex **W = cmatrix(0, M - 1, 0, M - 1), **G;
  for (int i = 0; i < M; i++) for (int j = 0; j < M; j++) W[i][j] = W0[i][j];
  for (int i = 0; i < M; i++) { W[i][i].r = E + W[i][i].r; W[i][i].i += eta; }
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
    W[3*N+i][3*N+j].r -= sig[i][j].r; W[3*N+i][3*N+j].i -= sig[i][j].i; }
  G = cmatinv(W, M);
  double r = 0, m = 0;
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
    double fr = 0, fi = 0;
    for (int k = 0; k < N; k++) for (int l = 0; l < N; l++) {
      vides_complex b = B[3*N+i][k], g = G[k][l], c = BD[l][3*N+j];
      double tr = b.r*g.r - b.i*g.i, ti = b.r*g.i + b.i*g.r;
      fr += tr*c.r - ti*c.i; fi += tr*c.i + ti*c.r; }
    r = fmax(r, hypot(fr - sig[i][j].r, fi - sig[i][j].i));
    m = fmax(m, hypot(sig[i][j].r, sig[i][j].i)); }
  cfree_cmatrix(W, 0, M - 1, 0, M - 1); cfree_cmatrix(G, 0, M - 1, 0, M - 1);
  return m > 0 ? r / m : r;
}

/* GPU vs CPU decimation on one lead cell, at NB energies. */
static int check_decim_cell(const char *label, int N, vides_complex **W0,
                            vides_complex **B, vides_complex **BD,
                            double eta, double tol)
{
  const int M = 4 * N, NB = 9;
  double E[NB];
  vides_complex **cpu[NB], **gpu[NB];
  for (int b = 0; b < NB; b++) {
    E[b] = -2.43 + 0.61 * b;
    cpu[b] = cmatrix(0, N - 1, 0, N - 1);
    gpu[b] = cmatrix(0, N - 1, 0, N - 1);
  }
  size_t before = vides_emu_live_bytes();
  int rc_c = vides_decimation_batch_cpu(M, NB, E, eta, W0, B, BD, 3 * N, N, cpu);
  int rc_g = vides_decimation_batch_gpu(M, NB, E, eta, W0, B, BD, 3 * N, N, gpu);
  size_t after = vides_emu_live_bytes();

  double worst = 0, sc, rc_res = 0, rg_res = 0;
  int ok = (rc_c == 0 && rc_g == 0);
  if (ok)
    for (int b = 0; b < NB; b++) {
      double d = blk_diff(cpu[b], gpu[b], N, &sc);
      if (sc > 0 && d / sc > worst) worst = d / sc;
      rc_res = fmax(rc_res, lead_residual(cpu[b], E[b], eta, N, W0, B, BD));
      rg_res = fmax(rg_res, lead_residual(gpu[b], E[b], eta, N, W0, B, BD));
    }
  /* Pass if GPU and CPU agree to tol, or if the GPU result satisfies the
     lead equation about as well as the CPU's -- then the gap is the
     problem's conditioning (decimation at small eta), not a GPU bug. */
  int agree = worst <= tol, as_good = rg_res <= 10 * rc_res + 1e-12;
  ok = ok && (agree || as_good) && after == before;
  printf("  %-30s eta=%-6g  GPU vs CPU rel %-8.1e residual cpu %-8.1e gpu %-8.1e%s  %s\n",
         label, eta, worst, rc_res, rg_res, after != before ? "  LEAK" : "",
         ok ? "ok" : "MISMATCH");
  free_out(cpu, NB, N); free_out(gpu, NB, N);
  return ok;
}

static int check_decimation(void)
{
  int ok = 1;
  const int N = 6, M = 24;
  double Em1[4 * N];
  for (int i = 0; i < 4 * N; i++) Em1[i] = -0.2 + 0.05 * (i % 3);

  vides_complex **W0 = cmatrix(0, M - 1, 0, M - 1), **B = cmatrix(0, M - 1, 0, M - 1),
                **BD = cmatrix(0, M - 1, 0, M - 1);
  selfGNR_cell(Em1, N, -2.7, W0, B, BD);

  /* 1. The CPU batch must reproduce stock selfGNR() exactly -- it is what
        CPU-only GNR runs now go through. */
  {
    const int NB = 5;
    double E[NB] = {-2.1, -0.7, 0.3, 1.1, 2.6};
    vides_complex **out[NB];
    for (int b = 0; b < NB; b++) out[b] = cmatrix(0, N - 1, 0, N - 1);
    vides_decimation_batch_cpu(M, NB, E, 1e-5, W0, B, BD, 3 * N, N, out);
    int exact = 1;
    for (int b = 0; b < NB; b++) {
      vides_complex **ref = selfGNR(E[b], Em1, N, -2.7, 1e-5);
      for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
          if (ref[i][j].r != out[b][i][j].r || ref[i][j].i != out[b][i][j].i) exact = 0;
      cfree_cmatrix(ref, 0, N - 1, 0, N - 1);
    }
    printf("  %-30s CPU batch == stock selfGNR, bit for bit:  %s\n",
           "GNR self-energy", exact ? "ok" : "MISMATCH");
    ok &= exact;
    free_out(out, NB, N);
  }

  /* 2. GPU vs CPU decimation, GNR cell */
  ok &= check_decim_cell("GNR cell (n=6)", N, W0, B, BD, 1e-5, 1e-9);
  ok &= check_decim_cell("GNR cell (n=6)", N, W0, B, BD, 1e-3, 1e-9);

  /* 3. eta must reach the GPU: results at two etas must differ */
  {
    const int NB = 1;
    double E[NB] = {0.83};
    vides_complex **a[NB], **b[NB];
    a[0] = cmatrix(0, N - 1, 0, N - 1); b[0] = cmatrix(0, N - 1, 0, N - 1);
    vides_decimation_batch_gpu(M, NB, E, 1e-5, W0, B, BD, 3 * N, N, a);
    vides_decimation_batch_gpu(M, NB, E, 1e-1, W0, B, BD, 3 * N, N, b);
    double sc, d = blk_diff(a[0], b[0], N, &sc);
    int passes_eta = d / sc > 1e-3;
    printf("  %-30s eta changes the GPU result (rel %.2e):  %s\n",
           "broadening honoured", d / sc, passes_eta ? "ok" : "MISMATCH");
    ok &= passes_eta;
    free_out(a, NB, N); free_out(b, NB, N);
  }

  /* 4. GPU vs CPU decimation, Hamiltonian-path cell (random four-slice
        periodic lead, both contacts) */
  {
    const int Nc = 13;
    Problem p;
    build(p, N, Nc, 1, 0xC0FFEEULL);
    /* make the lead four-slice periodic, as selfH_dec_cell assumes */
    for (int s = 4; s < Nc; s++)
      for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
          p.diag[s][i][j] = p.diag[s % 4][i][j];
          if (s < Nc - 1) p.up[s][i][j] = p.up[s % 4][i][j];
          p.low[s][i][j] = p.low[(s - 1) % 4 + 1][i][j];
        }
    for (int lead = 0; lead < 2; lead++) {
      selfH_dec_cell(p.diag, p.up, p.low, N, Nc, lead, W0, B, BD);
      char lab[64];
      snprintf(lab, sizeof lab, "Hamiltonian cell, %s", lead ? "drain" : "source");
      ok &= check_decim_cell(lab, N, W0, B, BD, 1e-5, 1e-9);
      ok &= check_decim_cell(lab, N, W0, B, BD, 1e-3, 1e-9);
    }
    destroy(p);
  }

  cfree_cmatrix(W0, 0, M - 1, 0, M - 1);
  cfree_cmatrix(B, 0, M - 1, 0, M - 1);
  cfree_cmatrix(BD, 0, M - 1, 0, M - 1);
  return ok;
}

/* The block reduction in k_trace_prod is the one thing the serial
   emulation cannot exercise, so check the tree itself directly. */
static int check_reduction_tree(void)
{
  const int B = 128, n = 37;
  double sh[128], ref = 0.0, part[128];
  for (int t = 0; t < B; t++) {
    double s = 0.0;
    for (int i = t; i < n; i += B) s += 1.0/(i+1.0);
    part[t] = s;
  }
  for (int i = 0; i < n; i++) ref += 1.0/(i+1.0);
  memcpy(sh, part, sizeof sh);
  for (unsigned s2 = B/2u; s2 > 0u; s2 >>= 1)
    for (unsigned t = 0; t < s2; t++) sh[t] += sh[t + s2];
  int ok = fabs(sh[0] - ref) < 1e-12;
  printf("  reduction tree (blockDim=%d, n=%d): sum=%.15g ref=%.15g  %s\n",
         B, n, sh[0], ref, ok ? "ok" : "MISMATCH");
  return ok;
}

/* The support functions around the driver: device probe, batch sizing and
   the dispatcher's chunk choice.  vides_gpu_batch_size() is where the
   operator-precedence bug lived, so its output is checked for sanity, not
   just for not crashing. */
static int check_sizing(void)
{
  int ok = 1;
  printf("  backend: %s\n", vides_gpu_describe());
  if (!vides_gpu_compiled()) { printf("    vides_gpu_compiled() = 0\n"); ok = 0; }
  if (!vides_gpu_available()) { printf("    vides_gpu_available() = 0\n"); ok = 0; }

  struct { int n, Nc; const char *what; } shapes[] = {
    { 20, 100, "CNT-ish   " }, { 24, 400, "GNR-ish   " },
    { 64, 200, "mid       " }, { 400, 100, "nanowire  " },
  };
  for (unsigned i = 0; i < sizeof shapes/sizeof shapes[0]; i++) {
    vides_rgf_desc d;
    memset(&d, 0, sizeof d);
    d.n = shapes[i].n; d.Nc = shapes[i].Nc; d.NB = 1;
    d.variant = VIDES_RGF_STD; d.flagtrans = 1; d.eta = 1e-3;
    int nb = vides_gpu_batch_size(&d);
    int ch = vides_negf_chunk(&d);
    /* 4 GB free is reported by the emulator, so every one of these shapes
       must fit many energies; 0 or 1 would mean the sizing collapsed. */
    int good = (nb >= 2 && nb <= 1024 && ch >= 1 && ch <= nb);
    printf("    %s n=%-4d Nc=%-4d  batch=%-5d chunk=%-5d  %s\n",
           shapes[i].what, d.n, d.Nc, nb, ch, good ? "ok" : "SUSPECT");
    if (!good) ok = 0;
  }
  return ok;
}

int main(void)
{
  int ok = 1;
  printf("GPU driver vs stock CPU path, over the emulated CUDA runtime\n");
  printf("============================================================\n");

  ok &= run_case("STD  (LDOS / CNT, GNR)", VIDES_RGF_STD,  6, 5, 4, 0, NULL, 1e-9);
  ok &= run_case("LAKE (LDOS_Lake / H, nanowire)", VIDES_RGF_LAKE, 6, 5, 4, 0, NULL, 1e-9);

  int order[3] = {0, 1, 5};
  ok &= run_case("MODE (LDOSMODE / CNT modes)", VIDES_RGF_MODE, 3, 5, 4, 6, order, 1e-9);

  /* A single block and a single energy are the degenerate shapes most
     likely to be mis-indexed. */
  ok &= run_case("STD  Nc=2 NB=1 (edge case)", VIDES_RGF_STD, 4, 2, 1, 0, NULL, 1e-9);
  ok &= run_case("LAKE Nc=2 NB=1 (edge case)", VIDES_RGF_LAKE, 4, 2, 1, 0, NULL, 1e-9);
  ok &= run_case("STD  larger (n=12 Nc=9 NB=7)", VIDES_RGF_STD, 12, 9, 7, 0, NULL, 1e-9);

  ok &= check_reduction_tree();
  printf("  ---- contact self-energy (batched decimation)\n");
  ok &= check_decimation();
  printf("  ----\n");
  ok &= check_sizing();

  printf("============================================================\n");
  printf("peak emulated device memory: %.2f MB\n",
         (double)vides_emu_peak_bytes()/1048576.0);
  printf("%s\n", ok ? "ALL GPU DRIVER CHECKS PASSED" : "FAILURES PRESENT");
  return ok ? 0 : 1;
}
