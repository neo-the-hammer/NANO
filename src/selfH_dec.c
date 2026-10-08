// ======================================================================
//  See selfH_dec.h.
//
//  This file is released under the BSD license, as the rest of ViDES.
//  See "license.txt".
// ======================================================================
#include <stdlib.h>
#include "selfH_dec.h"
#include "cmatrix.h"
#include "cfree_cmatrix.h"
#include "Gzerozero.h"

/* Copy an N x N block into position (bi,bj) of a 4N x 4N matrix. */
static void put_block(complex **M, int N, int bi, int bj, complex **B)
{
  int i, j;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      M[bi * N + i][bj * N + j] = B[i][j];
}

void selfH_dec_cell(complex ***diag, complex ***updiag, complex ***lowdiag,
                    int N, int Nc, int lead,
                    complex **W0, complex **BETA, complex **BETADAGA)
{
  int i, j, k, M = 4 * N;

  for (i = 0; i < M; i++)
    for (j = 0; j < M; j++) {
      W0[i][j].r = W0[i][j].i = 0.0;
      BETA[i][j].r = BETA[i][j].i = 0.0;
      BETADAGA[i][j].r = BETADAGA[i][j].i = 0.0;
    }

  if (lead == 1) {
    /* Drain.  Block k is slice Nc-4+k; the lead continues to the right,
       its first slice being of the same type as Nc-4. */
    for (k = 0; k < 4; k++) put_block(W0, N, k, k, diag[Nc - 4 + k]);
    for (k = 0; k < 3; k++) {
      put_block(W0, N, k, k + 1, updiag[Nc - 4 + k]);   /* s -> s+1 */
      put_block(W0, N, k + 1, k, lowdiag[Nc - 3 + k]);  /* s+1 -> s */
    }
    put_block(BETA,     N, 3, 0, updiag[Nc - 5]);  /* Nc-1 -> Nc ~ Nc-5 -> Nc-4 */
    put_block(BETADAGA, N, 0, 3, lowdiag[Nc - 4]); /* Nc -> Nc-1 ~ Nc-4 -> Nc-5 */
  } else {
    /* Source, mirrored so the lead again continues "to the right":
       block k is slice 3-k, and moving from block k to k+1 is moving
       from slice 3-k to 2-k.  Orbital order inside each slice is left as
       is, so no flip of the result is needed (selfH_new() flips the whole
       cell and then flips the self-energy back). */
    for (k = 0; k < 4; k++) put_block(W0, N, k, k, diag[3 - k]);
    for (k = 0; k < 3; k++) {
      put_block(W0, N, k, k + 1, lowdiag[3 - k]);   /* slice 3-k -> 2-k */
      put_block(W0, N, k + 1, k, updiag[2 - k]);    /* slice 2-k -> 3-k */
    }
    put_block(BETA,     N, 3, 0, lowdiag[4]);  /* 0 -> -1 ~ 4 -> 3 */
    put_block(BETADAGA, N, 0, 3, updiag[3]);   /* -1 -> 0 ~ 3 -> 4 */
  }
}

complex **selfH_dec(double E, complex ***diag, complex ***updiag,
                    complex ***lowdiag, int N, int Nc, int lead, double eta)
{
  int i, j, M = 4 * N;
  complex **W = cmatrix(0, M - 1, 0, M - 1);
  complex **B = cmatrix(0, M - 1, 0, M - 1);
  complex **BD = cmatrix(0, M - 1, 0, M - 1);
  complex **Gz, **sigma = cmatrix(0, N - 1, 0, N - 1);

  selfH_dec_cell(diag, updiag, lowdiag, N, Nc, lead, W, B, BD);
  /* Same diagonal as selfH_new(): (E + Re d) + i (Im d + eta). */
  for (i = 0; i < M; i++) {
    W[i][i].r = E + W[i][i].r;
    W[i][i].i = W[i][i].i + eta;
  }

  Gz = Gzerozero(W, B, BD, M);
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      sigma[i][j] = Gz[3 * N + i][3 * N + j];

  cfree_cmatrix(Gz, 0, M - 1, 0, M - 1);
  cfree_cmatrix(W, 0, M - 1, 0, M - 1);
  cfree_cmatrix(B, 0, M - 1, 0, M - 1);
  cfree_cmatrix(BD, 0, M - 1, 0, M - 1);
  return sigma;
}
