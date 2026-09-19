/* Fortran-ABI zgemm_/zgetrf_/zgetri_ for the emulation test, so the stock
 * ViDES routines (cmatmul.c, cmatinv.c) can run without a real LAPACK.
 * Not part of the shipped library. */
#include <stdlib.h>
#include <string.h>
#include "emu_linalg.h"

void zgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K,
            emu_cplx *ALPHA, emu_cplx *A, int *LDA, emu_cplx *B, int *LDB,
            emu_cplx *BETA, emu_cplx *C, int *LDC)
{
  int ca = (*TRANSA == 'C' || *TRANSA == 'c');
  int cb = (*TRANSB == 'C' || *TRANSB == 'c');
  emu_gemm(ca, cb, *M, *N, *K, *ALPHA, A, *LDA, B, *LDB, *BETA, C, *LDC);
}

void zgetrf_(int *M, int *N, emu_cplx *A, int *LDA, int *IPIV, int *INFO)
{
  *INFO = emu_getrf(A, *N, *LDA, IPIV);   /* square use only */
  (void)M;
}

void zgetri_(int *N, emu_cplx *A, int *LDA, int *IPIV,
             emu_cplx *WORK, int *LWORK, int *INFO)
{
  int n = *N;
  emu_cplx *C = (emu_cplx *)malloc((size_t)n*n*sizeof(emu_cplx));
  (void)WORK; (void)LWORK;
  if (!C) { *INFO = -1; return; }
  *INFO = emu_getri(A, n, *LDA, IPIV, C, n);
  memcpy(A, C, (size_t)n*n*sizeof(emu_cplx));
  free(C);
}
