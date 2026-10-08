// ======================================================================
//  Copyright (c) 2004-2010, G. Fiori, G. Iannaccone  University of Pisa
//  
//  This file is released under the BSD license.
//  See the file "license.txt" for information on usage and
//  redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
// ====================================================================== 
#include "Gzerozero.h"
#include <math.h>
#include <stdio.h>
// In wmH you put the EI-H, where H is the Hamiltonian, and 
// E is the energy. In BETA and BETADAGA you put the 
// hopping terms, i.e. the up and downdiagonal elements of the
// hamiltonian. OUTPUT : BETA gs BETADAGA, with gs the surface Green's
// function of the semi-infinite lead.
//
// gs is computed with the Sancho-Rubio decimation (J. Phys. F 15, 851
// (1985)).  The original code used the transfer-matrix form of the same
// method (J. Phys. F 14, 1205 (1984)), T = t0 + t~0 t1 + t~0 t~1 t2 + ...,
// gs = (wmH - BETA T)^-1.  That form is exact in exact arithmetic, but one
// of t_i, t~_i can grow while the other underflows: their product stays
// finite, the factors overflow, and T picks up inf * 0 = NaN.  Graphene
// ribbon leads hit this at ordinary energies.  In the Sancho-Rubio form
// both couplings shrink together and the result only accumulates finite
// corrections:
//
//   g    = A^-1
//   As  <- As - a g b                 (surface cell, renormalised)
//   A   <- A  - a g b - b g a         (bulk cell, renormalised)
//   a   <- a g a,   b <- b g b        (couplings across 2^k cells)
//
// from A = As = wmH, a = BETA, b = BETADAGA, until a and b are negligible;
// then gs = As^-1.  Stopping rule: every element of a and b below
// VIDES_DECIM_TOL times the largest element of BETA (the remaining
// correction is quadratic in that), or VIDES_DECIM_MAXIT iterations.
// vides_gpu.cu runs the same recursion and the same stopping rule.

static double cmaxabs(complex **X, int N)
{
  double m = 0, v;
  int i, j;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++) {
      v = fabs(X[i][j].r) + fabs(X[i][j].i);
      if (!(v <= m)) m = v;          /* also propagates NaN */
    }
  return m;
}

static void cmatcopy(complex **dst, complex **src, int N)
{
  int i, j;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      dst[i][j] = src[i][j];
}

complex **Gzerozero(complex **wmH,complex **BETA,complex **BETADAGA,int N)
{
  complex **A, **As, **a, **b, **g, **ga, **gb, **P, **Q, **an, **bn, **gs, **Gz;
  double thr;
  int i, j, it;

  A  = cmatrix(0, N-1, 0, N-1);
  As = cmatrix(0, N-1, 0, N-1);
  a  = cmatrix(0, N-1, 0, N-1);
  b  = cmatrix(0, N-1, 0, N-1);
  cmatcopy(A, wmH, N);
  cmatcopy(As, wmH, N);
  cmatcopy(a, BETA, N);
  cmatcopy(b, BETADAGA, N);
  thr = VIDES_DECIM_TOL * cmaxabs(BETA, N);

  for (it = 0; it < VIDES_DECIM_MAXIT; it++) {
    double ma = cmaxabs(a, N), mb = cmaxabs(b, N);
    if (ma <= thr && mb <= thr) break;
    if (!(ma < HUGE_VAL) || !(mb < HUGE_VAL)) break;   /* inf/NaN: give up */

    g  = cmatinv(A, N);
    gb = cmatmul(g, b, N);          /* g b */
    ga = cmatmul(g, a, N);          /* g a */
    P  = cmatmul(a, gb, N);         /* a g b */
    Q  = cmatmul(b, ga, N);         /* b g a */
    an = cmatmul(a, ga, N);         /* a g a */
    bn = cmatmul(b, gb, N);         /* b g b */
    for (i = 0; i < N; i++)
      for (j = 0; j < N; j++) {
        As[i][j].r -= P[i][j].r;  As[i][j].i -= P[i][j].i;
        A[i][j].r  -= P[i][j].r + Q[i][j].r;
        A[i][j].i  -= P[i][j].i + Q[i][j].i;
      }
    cmatcopy(a, an, N);
    cmatcopy(b, bn, N);
    cfree_cmatrix(g, 0, N-1, 0, N-1);  cfree_cmatrix(gb, 0, N-1, 0, N-1);
    cfree_cmatrix(ga, 0, N-1, 0, N-1); cfree_cmatrix(P, 0, N-1, 0, N-1);
    cfree_cmatrix(Q, 0, N-1, 0, N-1);  cfree_cmatrix(an, 0, N-1, 0, N-1);
    cfree_cmatrix(bn, 0, N-1, 0, N-1);
  }

  gs = cmatinv(As, N);
  Gz = cmatmul3(BETA, gs, BETADAGA, N);

  cfree_cmatrix(A, 0, N-1, 0, N-1);
  cfree_cmatrix(As, 0, N-1, 0, N-1);
  cfree_cmatrix(a, 0, N-1, 0, N-1);
  cfree_cmatrix(b, 0, N-1, 0, N-1);
  cfree_cmatrix(gs, 0, N-1, 0, N-1);
  return Gz;
}
