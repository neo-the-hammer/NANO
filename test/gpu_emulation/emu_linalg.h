/* Reference complex linear algebra for the GPU emulation test.
 *
 * Column-major throughout, matching both LAPACK and cuBLAS.  These are
 * plain textbook implementations -- slow, but simple enough to be read
 * and trusted, which is the point: they stand in for LAPACK and cuBLAS so
 * that vides_gpu.cu can be executed and checked on a machine with neither.
 *
 * Not part of the shipped library.
 */
#ifndef EMU_LINALG_H
#define EMU_LINALG_H

#include <math.h>
#include <string.h>

typedef struct { double r, i; } emu_cplx;

static emu_cplx emu_mul(emu_cplx a, emu_cplx b)
{ emu_cplx c; c.r = a.r*b.r - a.i*b.i; c.i = a.r*b.i + a.i*b.r; return c; }

static emu_cplx emu_sub(emu_cplx a, emu_cplx b)
{ emu_cplx c; c.r = a.r-b.r; c.i = a.i-b.i; return c; }

static emu_cplx emu_div(emu_cplx a, emu_cplx b)
{
  double d = b.r*b.r + b.i*b.i;
  emu_cplx c;
  c.r = (a.r*b.r + a.i*b.i)/d;
  c.i = (a.i*b.r - a.r*b.i)/d;
  return c;
}

static double emu_abs2(emu_cplx a) { return a.r*a.r + a.i*a.i; }

/* LU with partial pivoting, column-major, in place.  A = P*L*U with L unit
 * lower.  ipiv is 1-based, LAPACK style: at step k, row k was swapped with
 * row ipiv[k]-1.  Returns 0, or k+1 if U(k,k) came out exactly zero.
 */
static int emu_getrf(emu_cplx *A, int n, int lda, int *ipiv)
{
  int k, i, j, p, info = 0;
  for (k = 0; k < n; k++) {
    double best = -1.0;
    p = k;
    for (i = k; i < n; i++) {
      double m = emu_abs2(A[i + (size_t)k*lda]);
      if (m > best) { best = m; p = i; }
    }
    ipiv[k] = p + 1;
    if (best == 0.0) { if (!info) info = k + 1; continue; }
    if (p != k)
      for (j = 0; j < n; j++) {
        emu_cplx t = A[k + (size_t)j*lda];
        A[k + (size_t)j*lda] = A[p + (size_t)j*lda];
        A[p + (size_t)j*lda] = t;
      }
    for (i = k+1; i < n; i++) {
      emu_cplx m = emu_div(A[i + (size_t)k*lda], A[k + (size_t)k*lda]);
      A[i + (size_t)k*lda] = m;
      for (j = k+1; j < n; j++)
        A[i + (size_t)j*lda] =
          emu_sub(A[i + (size_t)j*lda], emu_mul(m, A[k + (size_t)j*lda]));
    }
  }
  return info;
}

/* Inverse from an emu_getrf factorisation: solve A x = e_j column by
 * column.  LU may alias nothing in C.
 */
static int emu_getri(const emu_cplx *LU, int n, int ldlu, const int *ipiv,
                     emu_cplx *C, int ldc)
{
  int j, k, i;
  emu_cplx *b = (emu_cplx *)malloc((size_t)n * sizeof(emu_cplx));
  if (!b) return -1;

  for (j = 0; j < n; j++) {
    for (i = 0; i < n; i++) { b[i].r = (i == j) ? 1.0 : 0.0; b[i].i = 0.0; }
    /* inv(P) b */
    for (k = 0; k < n; k++) {
      int p = ipiv[k] - 1;
      if (p != k) { emu_cplx t = b[k]; b[k] = b[p]; b[p] = t; }
    }
    /* forward solve L y = b, L unit lower */
    for (k = 0; k < n; k++)
      for (i = k+1; i < n; i++)
        b[i] = emu_sub(b[i], emu_mul(LU[i + (size_t)k*ldlu], b[k]));
    /* back solve U x = y */
    for (k = n-1; k >= 0; k--) {
      for (i = k+1; i < n; i++)
        b[k] = emu_sub(b[k], emu_mul(LU[k + (size_t)i*ldlu], b[i]));
      b[k] = emu_div(b[k], LU[k + (size_t)k*ldlu]);
    }
    for (i = 0; i < n; i++) C[i + (size_t)j*ldc] = b[i];
  }
  free(b);
  return 0;
}

/* General column-major gemm supporting the 'N' and conjugate-transpose
 * 'C' operations, which is all ViDES and vides_gpu.cu ask for.
 *   opA is m x k, opB is k x n, C is m x n.
 */
static void emu_gemm(int conjA, int conjB, int m, int n, int k,
                     emu_cplx alpha,
                     const emu_cplx *A, int lda,
                     const emu_cplx *B, int ldb,
                     emu_cplx beta,
                     emu_cplx *C, int ldc)
{
  int i, j, l;
  for (j = 0; j < n; j++)
    for (i = 0; i < m; i++) {
      emu_cplx s; s.r = 0.0; s.i = 0.0;
      for (l = 0; l < k; l++) {
        emu_cplx a = conjA ? A[l + (size_t)i*lda] : A[i + (size_t)l*lda];
        emu_cplx b = conjB ? B[j + (size_t)l*ldb] : B[l + (size_t)j*ldb];
        if (conjA) a.i = -a.i;
        if (conjB) b.i = -b.i;
        emu_cplx t = emu_mul(a, b);
        s.r += t.r; s.i += t.i;
      }
      s = emu_mul(alpha, s);
      if (beta.r != 0.0 || beta.i != 0.0) {
        emu_cplx c = emu_mul(beta, C[i + (size_t)j*ldc]);
        s.r += c.r; s.i += c.i;
      }
      C[i + (size_t)j*ldc] = s;
    }
}

#endif
