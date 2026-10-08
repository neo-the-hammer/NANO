// ======================================================================
//  Contact self-energy for the generic Hamiltonian path (H_charge_T) by
//  Lopez-Sancho decimation, as an alternative to selfH_new()'s
//  eigen-decomposition (mode matching).
//
//  Both treat the contact as a semi-infinite lead that repeats the
//  outermost four slices of the device, and in exact arithmetic give the
//  same self-energy.  Decimation needs only inversions and products, so
//  it batches over energies on the GPU; selfH_new() needs a general
//  non-symmetric eigensolver, which CUDA 11 does not provide.
//
//  This file is released under the BSD license, as the rest of ViDES.
//  See "license.txt".
// ======================================================================
#ifndef SELFH_DEC_H
#define SELFH_DEC_H

#include "complex.h"

/* Build the energy-independent part of the 4N x 4N lead cell for the
   decimation, in the form Gzerozero() takes:

     wmH(E) = W0 + (E + i*eta) I,   BETA, BETADAGA (cell-to-cell coupling)

   lead = 1: drain, cells of slices Nc-4 .. Nc-1 repeating to the right.
   lead = 0: source, cells of slices 3 .. 0 (mirrored) repeating to the
             left.
   The inter-cell coupling follows from four-slice periodicity:
   updiag[Nc-5] / lowdiag[Nc-4] for the drain, lowdiag[4] / updiag[3] for
   the source -- the same blocks selfH_new() uses.

   W0, BETA and BETADAGA must be 4N x 4N cmatrix()es; they are overwritten.
   The self-energy on the boundary slice is block (3,3) of Gzerozero()'s
   result.  Needs Nc >= 5. */
void selfH_dec_cell(complex ***diag, complex ***updiag, complex ***lowdiag,
                    int N, int Nc, int lead,
                    complex **W0, complex **BETA, complex **BETADAGA);

/* Drop-in replacement for selfH_new(): same arguments, same N x N result
   (a cmatrix() the caller frees). */
complex **selfH_dec(double E, complex ***diag, complex ***updiag,
                    complex ***lowdiag, int N, int Nc, int lead, double eta);

#endif
