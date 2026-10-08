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

#include "vides_rgf_batch.h"
#include "nrutil.h"
#include "LDOS.h"
#include "LDOS_Lake.h"
#include "LDOSmode.h"
#include "Gzerozero.h"
#include "cmatrix.h"
#include "cfree_cmatrix.h"

int vides_rgf_batch_cpu(const vides_rgf_desc *desc,
                        const double *E,
                        vides_complex ***diag,
                        vides_complex ***updiag,
                        vides_complex ***lowdiag,
                        vides_complex ***sigmas,
                        vides_complex ***sigmad,
                        double *A1, double *A2, double *T)
{
  const int n    = desc->n;
  const int Nc   = desc->Nc;
  const int NB   = desc->NB;
  const int nout = (desc->variant == VIDES_RGF_MODE) ? desc->Nreal : n;
  int b, i, j;

  if (n <= 0 || Nc <= 0 || NB <= 0) return 1;

  for (b = 0; b < NB; b++) {
    double **a1 = NULL, **a2 = NULL;
    double  t   = 0.0;

    switch (desc->variant) {
      case VIDES_RGF_LAKE:
        LDOS_Lake(E[b], lowdiag, diag, updiag, &a1, &a2,
                  sigmas[b], sigmad[b],
                  n, Nc, desc->flagtrans, &t, 0.0, desc->eta);
        break;
      case VIDES_RGF_MODE:
        LDOSMODE(E[b], lowdiag, diag, updiag, &a1, &a2,
                 sigmas[b], sigmad[b],
                 n, Nc, desc->flagtrans, &t,
                 desc->Nreal, (int *)desc->order, 0.0, desc->eta);
        break;
      case VIDES_RGF_STD:
      default:
        LDOS(E[b], lowdiag, diag, updiag, &a1, &a2,
             sigmas[b], sigmad[b],
             n, Nc, desc->flagtrans, &t, 0.0, desc->eta);
        break;
    }

    if (!a1 || !a2) return 1;

    for (i = 0; i < Nc; i++)
      for (j = 0; j < nout; j++) {
        A1[((size_t)b * Nc + i) * nout + j] = a1[i][j];
        A2[((size_t)b * Nc + i) * nout + j] = a2[i][j];
      }
    if (desc->flagtrans && T) T[b] = t;

    free_dmatrix(a1, 0, Nc - 1, 0, nout - 1);
    free_dmatrix(a2, 0, Nc - 1, 0, nout - 1);
  }
  return 0;
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
    int rc = vides_rgf_batch_gpu(desc, E, diag, updiag, lowdiag,
                                 sigmas, sigmad, A1, A2, T);
    if (rc == 0) return 0;
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

int vides_decimation_batch_cpu(int M, int NB, const double *E, double eta,
                               vides_complex **W0, vides_complex **BETA,
                               vides_complex **BETADAGA, int off, int nout,
                               vides_complex ***out)
{
  vides_complex **W = cmatrix(0, M - 1, 0, M - 1), **Gz;
  int b, i, j;

  for (b = 0; b < NB; b++) {
    for (i = 0; i < M; i++)
      for (j = 0; j < M; j++)
        W[i][j] = W0[i][j];
    /* Same diagonal the per-energy builders produce: E + Re, Im + eta. */
    for (i = 0; i < M; i++) {
      W[i][i].r = E[b] + W[i][i].r;
      W[i][i].i = W[i][i].i + eta;
    }
    Gz = Gzerozero(W, BETA, BETADAGA, M);
    for (i = 0; i < nout; i++)
      for (j = 0; j < nout; j++)
        out[b][i][j] = Gz[off + i][off + j];
    cfree_cmatrix(Gz, 0, M - 1, 0, M - 1);
  }
  cfree_cmatrix(W, 0, M - 1, 0, M - 1);
  return 0;
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
    int rc = vides_decimation_batch_gpu(M, NB, E, eta, W0, BETA, BETADAGA,
                                        off, nout, out);
    if (rc == 0) return 0;
    fprintf(stderr,
            "[ViDES/GPU] decimation batch failed (rc=%d), falling back to CPU\n",
            rc);
  }
  return vides_decimation_batch_cpu(M, NB, E, eta, W0, BETA, BETADAGA,
                                    off, nout, out);
}

int vides_selfh_use_decimation(double eta)
{
  const char *env = getenv("VIDES_SELFH");
  if (env && (env[0] == 'e' || env[0] == 'E')) return 0;
  if (env && (env[0] == 'd' || env[0] == 'D')) return 1;
  /* Below this broadening the eigen method is much the more accurate of
     the two (residual ~1e-12 against up to ~1e-6 for decimation at
     eta = 1e-8, at band edges), and it costs about as much per energy as
     one batched GPU decimation, so it stays the default there even on a
     GPU; the NEGF solve itself still runs on the GPU. */
  if (eta < VIDES_SELFH_DEC_MIN_ETA) return 0;
  return vides_gpu_available();
}
