// ======================================================================
//  Reference (CPU) implementation of the energy-batched RGF driver, and
//  the dispatcher that chooses between it and the CUDA backend.
//
//  The reference path deliberately does NOT reimplement the recursion.
//  It calls LDOS() / LDOS_Lake() / LDOSMODE() once per energy, exactly as
//  the scalar loop in CNT_charge_T.c and friends does today.  Two things
//  follow from that:
//
//    - Running an entry point through this driver with the GPU disabled
//      reproduces the stock numbers bit for bit, so the restructuring of
//      the energy loop can be validated on its own, without a GPU.
//    - It is an exact oracle for the CUDA backend: any disagreement
//      beyond round-off is a bug in the GPU path.
//
//  This file is released under the BSD license, as the rest of ViDES.
//  See "license.txt".
// ======================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

#include "vides_rgf_batch.h"
#include "nrutil.h"
#include "LDOS.h"
#include "LDOS_Lake.h"
#include "LDOSmode.h"
#include "Gzerozero.h"
#include "cmatrix.h"
#include "cfree_cmatrix.h"

complex **selfH_new(double E, complex ***diag, complex ***updiag,
                    complex ***lowdiag, int N, int Nc, int lead, double eta);

/* ------------------------------------------------------------------ */
/* Policy and the thread pool                                          */
/* ------------------------------------------------------------------ */

static int env_flag(const char *name, int dflt)
{
  const char *e = getenv(name);
  if (!e || !*e) return dflt;
  return !(e[0] == '0' || e[0] == 'n' || e[0] == 'N' || e[0] == 'f' || e[0] == 'F');
}

static int nt = 0;

void vides_set_cpu_threads(int n)
{
  nt = n < 1 ? 1 : (n > 256 ? 256 : n);
}

int vides_cpu_threads(void)
{
  if (!nt) {
    const char *e = getenv("VIDES_THREADS");
    long v = e ? strtol(e, NULL, 10) : 0;
    if (v < 1) v = sysconf(_SC_NPROCESSORS_ONLN);
    if (v < 1) v = 1;
    if (v > 256) v = 256;
    nt = (int)v;
  }
  return nt;
}

int vides_hybrid_enabled(void)
{
  return vides_gpu_available() && env_flag("VIDES_HYBRID", 1);
}

int vides_pipeline_enabled(void)
{
  return vides_gpu_available() && env_flag("VIDES_PIPELINE", 1);
}

typedef struct {
  vides_index_fn fn;
  void *ctx;
  int N;
  int next;            /* next index to hand out (atomic) */
} pf_state;

static void *pf_worker(void *arg)
{
  pf_state *p = (pf_state *)arg;
  int i;
  while ((i = __sync_fetch_and_add(&p->next, 1)) < p->N)
    p->fn(p->ctx, i);
  return NULL;
}

static int pf_primed = 0;

void vides_parallel_for(int N, int nthreads, vides_index_fn fn, void *ctx)
{
  pf_state p;
  pthread_t th[256];
  int k, nth = 0, start = 0;

  if (N <= 0) return;
  if (!__sync_lock_test_and_set(&pf_primed, 1)) {
    /* One index alone first: one-time initialisation inside LAPACK
       (dlamch's SAVEd machine constants) must not race. */
    fn(ctx, 0);
    start = 1;
  }
  if (nthreads > N - start) nthreads = N - start;
  if (nthreads > 256) nthreads = 256;
  if (nthreads <= 1) {
    for (k = start; k < N; k++) fn(ctx, k);
    return;
  }
  p.fn = fn; p.ctx = ctx; p.N = N; p.next = start;
  for (k = 1; k < nthreads; k++)
    if (pthread_create(&th[nth], NULL, pf_worker, &p) == 0) nth++;
  pf_worker(&p);                      /* the caller works too */
  for (k = 0; k < nth; k++) pthread_join(th[k], NULL);
}

typedef struct {
  const vides_rgf_desc *desc;
  const double *E;
  vides_complex ***diag, ***updiag, ***lowdiag, ***sigmas, ***sigmad;
  double *A1, *A2, *T;
  int err;
} rgf_ctx;

/* One energy: the stock scalar routine, results into slot b. */
static void rgf_one(void *arg, int b)
{
  rgf_ctx *c = (rgf_ctx *)arg;
  const vides_rgf_desc *desc = c->desc;
  const int n    = desc->n;
  const int Nc   = desc->Nc;
  const int nout = (desc->variant == VIDES_RGF_MODE) ? desc->Nreal : n;
  double **a1 = NULL, **a2 = NULL;
  double  t   = 0.0;
  int i, j;

  switch (desc->variant) {
    case VIDES_RGF_LAKE:
      LDOS_Lake(c->E[b], c->lowdiag, c->diag, c->updiag, &a1, &a2,
                c->sigmas[b], c->sigmad[b],
                n, Nc, desc->flagtrans, &t, 0.0, desc->eta);
      break;
    case VIDES_RGF_MODE:
      LDOSMODE(c->E[b], c->lowdiag, c->diag, c->updiag, &a1, &a2,
               c->sigmas[b], c->sigmad[b],
               n, Nc, desc->flagtrans, &t,
               desc->Nreal, (int *)desc->order, 0.0, desc->eta);
      break;
    case VIDES_RGF_STD:
    default:
      LDOS(c->E[b], c->lowdiag, c->diag, c->updiag, &a1, &a2,
           c->sigmas[b], c->sigmad[b],
           n, Nc, desc->flagtrans, &t, 0.0, desc->eta);
      break;
  }

  if (!a1 || !a2) { c->err = 1; return; }

  for (i = 0; i < Nc; i++)
    for (j = 0; j < nout; j++) {
      c->A1[((size_t)b * Nc + i) * nout + j] = a1[i][j];
      c->A2[((size_t)b * Nc + i) * nout + j] = a2[i][j];
    }
  if (desc->flagtrans && c->T) c->T[b] = t;

  free_dmatrix(a1, 0, Nc - 1, 0, nout - 1);
  free_dmatrix(a2, 0, Nc - 1, 0, nout - 1);
}

/* Every energy runs the stock scalar code and writes only its own slot,
   so the result does not depend on the number of threads. */
int vides_rgf_batch_cpu(const vides_rgf_desc *desc,
                        const double *E,
                        vides_complex ***diag,
                        vides_complex ***updiag,
                        vides_complex ***lowdiag,
                        vides_complex ***sigmas,
                        vides_complex ***sigmad,
                        double *A1, double *A2, double *T)
{
  rgf_ctx c;

  if (desc->n <= 0 || desc->Nc <= 0 || desc->NB <= 0) return 1;
  c.desc = desc; c.E = E; c.diag = diag; c.updiag = updiag; c.lowdiag = lowdiag;
  c.sigmas = sigmas; c.sigmad = sigmad; c.A1 = A1; c.A2 = A2; c.T = T; c.err = 0;
  vides_parallel_for(desc->NB, vides_cpu_threads(), rgf_one, &c);
  return c.err;
}

/* ------------------------------------------------------------------ */
/* CPU + GPU co-execution                                              */
/* ------------------------------------------------------------------ */

/* Measured throughput (energies per second) of each side, per problem
   shape, smoothed over calls.  The CPU share of the next batch is
   rate_cpu / (rate_cpu + rate_gpu): both sides then finish together. */
typedef struct {
  int kind, size, Nc, var;
  double rc, rg, share;
} hy_entry;

#define HY_SLOTS 64
static hy_entry hy_tab[HY_SLOTS];
static int hy_used = 0, hy_next = 0;
static pthread_mutex_t hy_mx = PTHREAD_MUTEX_INITIALIZER;

static hy_entry *hy_find(int kind, int size, int Nc, int var, int create)
{
  int i;
  hy_entry *e;
  for (i = 0; i < hy_used; i++)
    if (hy_tab[i].kind == kind && hy_tab[i].size == size &&
        hy_tab[i].Nc == Nc && hy_tab[i].var == var) return &hy_tab[i];
  if (!create) return NULL;
  if (hy_used < HY_SLOTS) e = &hy_tab[hy_used++];
  else { e = &hy_tab[hy_next]; hy_next = (hy_next + 1) % HY_SLOTS; }
  e->kind = kind; e->size = size; e->Nc = Nc; e->var = var;
  e->rc = 0; e->rg = 0; e->share = -1;
  return e;
}

/* How many of NB energies go to the CPU. */
static int hy_cpu_count(int kind, int size, int Nc, int var, int NB)
{
  int nt = vides_cpu_threads(), ncpu;
  hy_entry *e;
  const char *fix = getenv("VIDES_HYBRID_SHARE");   /* fixed share, for experiments */

  if (NB < 2) return 0;
  if (fix && *fix) {
    double f = atof(fix);
    if (f < 0) f = 0;
    if (f > 1) f = 1;
    return (int)(f * NB + 0.5);
  }
  pthread_mutex_lock(&hy_mx);
  e = hy_find(kind, size, Nc, var, 1);
  if (e->rc <= 0 || e->rg <= 0) {
    /* No history yet: a small probe share, one or two energies per
       thread, so the first batch measures both sides. */
    ncpu = 2 * nt;
    if (ncpu > NB / 4) ncpu = NB / 4;
    if (ncpu < 1) ncpu = 1;
  } else {
    ncpu = (int)(NB * e->rc / (e->rc + e->rg) + 0.5);
  }
  pthread_mutex_unlock(&hy_mx);
  if (ncpu < 0) ncpu = 0;
  if (ncpu > NB) ncpu = NB;
  return ncpu;
}

static void hy_record(int kind, int size, int Nc, int var,
                      int ncpu, double tc, int ngpu, double tg, int NB)
{
  hy_entry *e;
  pthread_mutex_lock(&hy_mx);
  e = hy_find(kind, size, Nc, var, 1);
  if (ncpu > 0 && tc > 0) {
    double r = ncpu / tc;
    e->rc = e->rc > 0 ? 0.5 * e->rc + 0.5 * r : r;
  }
  if (ngpu > 0 && tg > 0) {
    double r = ngpu / tg;
    e->rg = e->rg > 0 ? 0.5 * e->rg + 0.5 * r : r;
  }
  e->share = NB > 0 ? (double)ncpu / NB : -1;
  pthread_mutex_unlock(&hy_mx);
  {
    const char *pr = getenv("VIDES_PROFILE");
    if (pr && pr[0] == '2')
      printf("[ViDES hybrid] %s size %d Nc %d: %d energies, CPU %d (%.1f ms), GPU %d (%.1f ms)\n",
             kind ? "decimation" : "rgf", size, Nc, NB, ncpu, 1e3 * tc, ngpu, 1e3 * tg);
  }
}

double vides_hybrid_cpu_share(int kind, int size, int Nc)
{
  hy_entry *e;
  double s = -1;
  pthread_mutex_lock(&hy_mx);
  e = hy_find(kind, size, Nc, -1, 0);
  if (!e) {
    int i;
    for (i = 0; i < hy_used; i++)
      if (hy_tab[i].kind == kind && hy_tab[i].size == size && hy_tab[i].Nc == Nc) { e = &hy_tab[i]; break; }
  }
  if (e) s = e->share;
  pthread_mutex_unlock(&hy_mx);
  return s;
}

/* A GPU batch on its own thread. */
struct vides_rgf_job {
  vides_rgf_desc d;
  const double *E;
  vides_complex ***diag, ***updiag, ***lowdiag, ***sigmas, ***sigmad;
  double *A1, *A2, *T;
  int rc, threaded, cpu_fallback;
  double t;
  pthread_t th;
};

static void *job_main(void *arg)
{
  vides_rgf_job *j = (vides_rgf_job *)arg;
  double t0 = vides_now();
  j->rc = vides_rgf_batch_gpu(&j->d, j->E, j->diag, j->updiag, j->lowdiag,
                              j->sigmas, j->sigmad, j->A1, j->A2, j->T);
  j->t = vides_now() - t0;
  if (j->rc != 0 && j->cpu_fallback) {
    fprintf(stderr, "[ViDES/GPU] batch failed (rc=%d), falling back to CPU\n", j->rc);
    j->rc = vides_rgf_batch_cpu(&j->d, j->E, j->diag, j->updiag, j->lowdiag,
                                j->sigmas, j->sigmad, j->A1, j->A2, j->T);
  }
  return NULL;
}

static vides_rgf_job *job_new(const vides_rgf_desc *desc, const double *E,
                              vides_complex ***diag, vides_complex ***updiag,
                              vides_complex ***lowdiag, vides_complex ***sigmas,
                              vides_complex ***sigmad, double *A1, double *A2,
                              double *T, int cpu_fallback)
{
  vides_rgf_job *j = (vides_rgf_job *)calloc(1, sizeof *j);
  if (!j) return NULL;
  j->d = *desc; j->E = E; j->diag = diag; j->updiag = updiag; j->lowdiag = lowdiag;
  j->sigmas = sigmas; j->sigmad = sigmad; j->A1 = A1; j->A2 = A2; j->T = T;
  j->cpu_fallback = cpu_fallback;
  j->threaded = (pthread_create(&j->th, NULL, job_main, j) == 0);
  if (!j->threaded) job_main(j);
  return j;
}

static int job_join(vides_rgf_job *j, double *t)
{
  int rc;
  if (!j) return -1;
  if (j->threaded) pthread_join(j->th, NULL);
  rc = j->rc;
  if (t) *t = j->t;
  free(j);
  return rc;
}

vides_rgf_job *vides_rgf_start(const vides_rgf_desc *desc, const double *E,
                               vides_complex ***diag, vides_complex ***updiag,
                               vides_complex ***lowdiag,
                               vides_complex ***sigmas, vides_complex ***sigmad,
                               double *A1, double *A2, double *T)
{
  return job_new(desc, E, diag, updiag, lowdiag, sigmas, sigmad, A1, A2, T, 1);
}

int vides_rgf_wait(vides_rgf_job *job)
{
  return job_join(job, NULL);
}

/* ------------------------------------------------------------------ */
/* Dispatcher                                                          */
/* ------------------------------------------------------------------ */

int vides_rgf_batch(const vides_rgf_desc *desc,
                    const double *E,
                    vides_complex ***diag,
                    vides_complex ***updiag,
                    vides_complex ***lowdiag,
                    vides_complex ***sigmas,
                    vides_complex ***sigmad,
                    double *A1, double *A2, double *T)
{
  if (vides_gpu_available()) {
    const int nout = (desc->variant == VIDES_RGF_MODE) ? desc->Nreal : desc->n;
    int ncpu = vides_hybrid_enabled()
               ? hy_cpu_count(0, desc->n, desc->Nc, desc->variant, desc->NB) : 0;
    int ngpu = desc->NB - ncpu, rc;

    if (ngpu <= 0)                     /* the CPU is the faster side here */
      {
        double t0 = vides_now();
        rc = vides_rgf_batch_cpu(desc, E, diag, updiag, lowdiag,
                                 sigmas, sigmad, A1, A2, T);
        hy_record(0, desc->n, desc->Nc, desc->variant, desc->NB,
                  vides_now() - t0, 0, 0, desc->NB);
        return rc;
      }
    if (ncpu == 0) {
      rc = vides_rgf_batch_gpu(desc, E, diag, updiag, lowdiag,
                               sigmas, sigmad, A1, A2, T);
      if (rc == 0) return 0;
    } else {
      /* GPU: energies [0, ngpu) on a background thread.  CPU threads:
         [ngpu, NB) meanwhile.  Disjoint outputs, so no locking. */
      vides_rgf_desc dg = *desc, dc = *desc;
      vides_rgf_job *job;
      size_t off = (size_t)ngpu * desc->Nc * nout;
      double tc, tg = 0, t0;
      int rcc;

      dg.NB = ngpu;
      dc.NB = ncpu;
      job = job_new(&dg, E, diag, updiag, lowdiag, sigmas, sigmad, A1, A2, T, 0);
      t0 = vides_now();
      rcc = vides_rgf_batch_cpu(&dc, E + ngpu, diag, updiag, lowdiag,
                                sigmas + ngpu, sigmad + ngpu, A1 + off, A2 + off,
                                T ? T + ngpu : NULL);
      tc = vides_now() - t0;
      rc = job_join(job, &tg);
      if (rcc) return rcc;
      if (rc == 0) {
        hy_record(0, desc->n, desc->Nc, desc->variant, ncpu, tc, ngpu, tg, desc->NB);
        return 0;
      }
      fprintf(stderr,
              "[ViDES/GPU] batch failed (rc=%d), falling back to CPU\n", rc);
      return vides_rgf_batch_cpu(&dg, E, diag, updiag, lowdiag,
                                 sigmas, sigmad, A1, A2, T);
    }
    /* A GPU failure is recoverable -- redo the batch on the CPU rather
       than losing the bias point.  Warn so it is not silent. */
    fprintf(stderr,
            "[ViDES/GPU] batch failed (rc=%d), falling back to CPU\n", rc);
  }
  return vides_rgf_batch_cpu(desc, E, diag, updiag, lowdiag,
                             sigmas, sigmad, A1, A2, T);
}

/* How many energies to handle per call.  On the GPU this is set by device
   memory; on the CPU a small chunk just amortises the per-call bookkeeping
   while keeping the temporary self-energy storage bounded. */
int vides_negf_chunk(const vides_rgf_desc *desc)
{
  const size_t nout = (size_t)((desc->variant == VIDES_RGF_MODE)
                               ? desc->Nreal : desc->n);
  size_t host_cap;
  int nb = 16;

  if (vides_gpu_available()) {
    int g = vides_gpu_batch_size(desc);
    if (g >= 1) nb = g;
  }

  /* The caller holds A1 and A2 for the whole chunk, so cap the batch so
     that pair stays under ~256 MB of host memory however much room the
     device reports. */
  host_cap = ((size_t)256 << 20)
             / (2 * (size_t)desc->Nc * (nout ? nout : 1) * sizeof(double));
  if (host_cap < 1) host_cap = 1;
  if ((size_t)nb > host_cap) nb = (int)host_cap;

  return nb < 1 ? 1 : nb;
}

/* ------------------------------------------------------------------ */
/* Profiling                                                           */
/* ------------------------------------------------------------------ */

double vides_now(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

void vides_profile_report(const char *who, double total,
                          double t_self, double t_solve)
{
  const char *env = getenv("VIDES_PROFILE");
  double other;
  if (!env || env[0] == '0') return;
  if (total <= 0) total = 1e-30;
  other = total - t_self - t_solve;
  printf("[ViDES profile] %s: total %.4f s = self-energy %.4f s (%.0f%%)"
         " + NEGF solve %.4f s (%.0f%%) + other %.4f s (%.0f%%)\n",
         who, total, t_self, 100.0 * t_self / total,
         t_solve, 100.0 * t_solve / total, other, 100.0 * other / total);
  fflush(stdout);
}

/* ------------------------------------------------------------------ */
/* Batched decimation                                                  */
/* ------------------------------------------------------------------ */

typedef struct {
  int M, off, nout;
  const double *E;
  double eta;
  vides_complex **W0, **BETA, **BETADAGA, ***out;
} dec_ctx;

static void dec_one(void *arg, int b)
{
  dec_ctx *c = (dec_ctx *)arg;
  const int M = c->M;
  vides_complex **W = cmatrix(0, M - 1, 0, M - 1), **Gz;
  int i, j;

  for (i = 0; i < M; i++)
    for (j = 0; j < M; j++)
      W[i][j] = c->W0[i][j];
  /* Same diagonal the per-energy builders produce: E + Re, Im + eta. */
  for (i = 0; i < M; i++) {
    W[i][i].r = c->E[b] + W[i][i].r;
    W[i][i].i = W[i][i].i + c->eta;
  }
  Gz = Gzerozero(W, c->BETA, c->BETADAGA, M);
  for (i = 0; i < c->nout; i++)
    for (j = 0; j < c->nout; j++)
      c->out[b][i][j] = Gz[c->off + i][c->off + j];
  cfree_cmatrix(Gz, 0, M - 1, 0, M - 1);
  cfree_cmatrix(W, 0, M - 1, 0, M - 1);
}

int vides_decimation_batch_cpu(int M, int NB, const double *E, double eta,
                               vides_complex **W0, vides_complex **BETA,
                               vides_complex **BETADAGA, int off, int nout,
                               vides_complex ***out)
{
  dec_ctx c;
  c.M = M; c.off = off; c.nout = nout; c.E = E; c.eta = eta;
  c.W0 = W0; c.BETA = BETA; c.BETADAGA = BETADAGA; c.out = out;
  vides_parallel_for(NB, vides_cpu_threads(), dec_one, &c);
  return 0;
}

typedef struct {
  int M, NB, off, nout, rc;
  const double *E;
  double eta, t;
  vides_complex **W0, **BETA, **BETADAGA, ***out;
} dec_job;

static void *dec_job_main(void *arg)
{
  dec_job *j = (dec_job *)arg;
  double t0 = vides_now();
  j->rc = vides_decimation_batch_gpu(j->M, j->NB, j->E, j->eta, j->W0, j->BETA,
                                     j->BETADAGA, j->off, j->nout, j->out);
  j->t = vides_now() - t0;
  return NULL;
}

int vides_decimation_batch(int M, int NB, const double *E, double eta,
                           vides_complex **W0, vides_complex **BETA,
                           vides_complex **BETADAGA, int off, int nout,
                           vides_complex ***out)
{
  int b;
  for (b = 0; b < NB; b++)
    out[b] = cmatrix(0, nout - 1, 0, nout - 1);

  if (vides_gpu_available()) {
    int ncpu = vides_hybrid_enabled() ? hy_cpu_count(1, M, 0, 0, NB) : 0;
    int ngpu = NB - ncpu, rc;
    if (ngpu <= 0) {
      double t0 = vides_now();
      vides_decimation_batch_cpu(M, NB, E, eta, W0, BETA, BETADAGA, off, nout, out);
      hy_record(1, M, 0, 0, NB, vides_now() - t0, 0, 0, NB);
      return 0;
    }
    if (ncpu == 0) {
      rc = vides_decimation_batch_gpu(M, NB, E, eta, W0, BETA, BETADAGA,
                                      off, nout, out);
      if (rc == 0) return 0;
    } else {
      dec_job j;
      pthread_t th;
      int threaded;
      double t0, tc;
      j.M = M; j.NB = ngpu; j.off = off; j.nout = nout; j.E = E; j.eta = eta;
      j.W0 = W0; j.BETA = BETA; j.BETADAGA = BETADAGA; j.out = out; j.rc = -1; j.t = 0;
      threaded = (pthread_create(&th, NULL, dec_job_main, &j) == 0);
      if (!threaded) dec_job_main(&j);
      t0 = vides_now();
      vides_decimation_batch_cpu(M, ncpu, E + ngpu, eta, W0, BETA, BETADAGA,
                                 off, nout, out + ngpu);
      tc = vides_now() - t0;
      if (threaded) pthread_join(th, NULL);
      rc = j.rc;
      if (rc == 0) {
        hy_record(1, M, 0, 0, ncpu, tc, ngpu, j.t, NB);
        return 0;
      }
      fprintf(stderr,
              "[ViDES/GPU] decimation batch failed (rc=%d), falling back to CPU\n", rc);
      return vides_decimation_batch_cpu(M, ngpu, E, eta, W0, BETA, BETADAGA,
                                        off, nout, out);
    }
    fprintf(stderr,
            "[ViDES/GPU] decimation batch failed (rc=%d), falling back to CPU\n",
            rc);
  }
  return vides_decimation_batch_cpu(M, NB, E, eta, W0, BETA, BETADAGA,
                                    off, nout, out);
}

/* ------------------------------------------------------------------ */
/* Threaded eigen self-energy                                          */
/* ------------------------------------------------------------------ */

typedef struct {
  const double *E;
  vides_complex ***diag, ***updiag, ***lowdiag, ***out;
  int N, Nc, lead;
  double eta;
} sh_ctx;

static void sh_one(void *arg, int b)
{
  sh_ctx *c = (sh_ctx *)arg;
  c->out[b] = selfH_new(c->E[b], c->diag, c->updiag, c->lowdiag,
                        c->N, c->Nc, c->lead, c->eta);
}

void vides_selfH_new_batch(int NB, const double *E, vides_complex ***diag,
                           vides_complex ***updiag, vides_complex ***lowdiag,
                           int N, int Nc, int lead, double eta,
                           vides_complex ***out)
{
  sh_ctx c;
  c.E = E; c.diag = diag; c.updiag = updiag; c.lowdiag = lowdiag; c.out = out;
  c.N = N; c.Nc = Nc; c.lead = lead; c.eta = eta;
  vides_parallel_for(NB, vides_cpu_threads(), sh_one, &c);
}

int vides_selfh_use_decimation(double eta)
{
  const char *env = getenv("VIDES_SELFH");
  if (env && (env[0] == 'e' || env[0] == 'E')) return 0;
  if (env && (env[0] == 'd' || env[0] == 'D')) return 1;
  /* Default: the eigen method, on the CPU threads, pipelined with the
     GPU solve (H_charge_T).  It is the more accurate of the two
     (residual ~1e-12 against up to ~1e-6 for decimation at small eta, A1)
     and, measured on a T4, no slower: one eigen self-energy costs about
     one batched GPU decimation per energy on a single core (S2), the
     pipeline hides it behind the GPU solve, and every further core divides
     it (Hamiltonian path 16x40: 2.0 s eigen vs 2.4 s decimation on Colab,
     S7; with 4 cores the GPU decimation lost to the CPU, 0.6x).  GPU
     decimation stays available with VIDES_SELFH=dec. */
  (void)eta;
  return 0;
}
