#!/bin/sh
# Compare the Hamiltonian path's two contact self-energies -- the stock
# selfH_new() (eigen-decomposition) and selfH_dec() (decimation, the one
# that runs on the GPU) -- on graphene-ribbon and random periodic leads,
# both contacts, two values of eta.  Also prints each method's residual
# against the lead's own fixed-point equation, a method-independent
# accuracy measure.  Needs no GPU and no LAPACK (uses lapack_shim.c).
#
#   sh test/gpu_emulation/run_selfenergy_check.sh
set -e
here=$(cd "$(dirname "$0")" && pwd)
src=$here/../../src
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

FILES="selfH_new selfH_dec Gzerozero create_updiagGNR create_lowdiagGNR create_beta1GNR \
       create_beta2GNR create_beta2transpGNR ccvector cdabs cfree_ccvector cfree_cmatrix \
       cmatinv cmatmul cmatmul3 cmatnorm2diff cmatrix cmatsub cmatsum cmatvect cmul \
       complass eigenvalues_non_symmetric_matrix flip_cmatrix nrutil"
for f in $FILES; do
  gcc -c -O1 -std=gnu89 -w -I"$src" "$src/$f.c" -o "$tmp/$f.o"
done
gcc -c -O1 -w -I"$here" "$here/lapack_shim.c" -o "$tmp/shim.o"
gcc -c -O1 -std=gnu99 -w -I"$src" -I"$here" "$here/selfenergy_check.c" -o "$tmp/main.o"
gcc -o "$tmp/check" "$tmp"/*.o -lm
"$tmp/check"
