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
    if (b < Nc-1) for (int i = 0; i < n; i++) p.up[b][i][i].r = -2.7;
    if (b > 0)    for (int i = 0; i < n; i++) p.low[b][i][i].r = -2.7;
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
  printf("  ----\n");
  ok &= check_sizing();

  printf("============================================================\n");
  printf("peak emulated device memory: %.2f MB\n",
         (double)vides_emu_peak_bytes()/1048576.0);
  printf("%s\n", ok ? "ALL GPU DRIVER CHECKS PASSED" : "FAILURES PRESENT");
  return ok ? 0 : 1;
}
