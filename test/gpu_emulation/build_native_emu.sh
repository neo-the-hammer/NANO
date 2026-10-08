#!/bin/sh
# Build experiments/native/vides_native.c against the emulated GPU, so the
# experiments suite can be exercised end to end on a machine with no GPU,
# no CUDA toolkit and no LAPACK:
#
#   sh test/gpu_emulation/build_native_emu.sh /tmp/vides_native_emu
#   VIDES_NATIVE_BIN=/tmp/vides_native_emu python3 experiments/run_all.py --only C1 C2
#
# The transform of vides_gpu.cu is the one run_test.sh uses.  Timings from
# this binary are meaningless (every "GPU" thread runs serially on the CPU);
# only the numerical comparisons are.
set -e
here=$(cd "$(dirname "$0")" && pwd)
src=$here/../../src
out=${1:-$here/vides_native_emu}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Reuse run_test.sh's launch helper and kernel-launch rewrite.
sed -n '/^cat > "\$tmp\/launch.h"/,/^EOF$/p' "$here/run_test.sh" > "$tmp/mk_launch.sh"
sed -n '/^perl -0777/,/vides_gpu.cpp"$/p' "$here/run_test.sh" > "$tmp/mk_gpu.sh"
here=$here src=$src tmp=$tmp sh -c ". '$tmp/mk_launch.sh'; . '$tmp/mk_gpu.sh'"
sed -i 's|#include "vides_rgf_batch.h"|#include "vides_rgf_batch.h"\n#include "launch.h"|' "$tmp/vides_gpu.cpp"

CFILES="nrutil cmatrix cmatrixm cvectorm ctensor4 c3tensor ccvector \
        cfree_cmatrix cfree_cvectorm cfree_ctensor4 cfree_c3tensor cfree_ccvector \
        complass csum csub cmatsub cmatsum cmatdaga cmatmul cmatmul3 cmatinv cmatRe \
        cdabs cIm cmatmul_proc spectralfun spectralfunmode transmission VAVdaga \
        rgfblock rgfblock_Lake LDOS LDOS_Lake LDOSmode vides_rgf_batch \
        Gzerozero cmatnorm2diff selfGNR selfH_dec selfH_new create_updiagGNR \
        create_lowdiagGNR create_beta1GNR create_beta2GNR create_beta2transpGNR \
        cmatvect cmul eigenvalues_non_symmetric_matrix flip_cmatrix"
OBJS=""
for f in $CFILES; do
  gcc -c -O2 -std=gnu89 -w -I"$src" -I"$here" "$src/$f.c" -o "$tmp/$f.o"
  OBJS="$OBJS $tmp/$f.o"
done
gcc -c -O2 -w -I"$here" "$here/lapack_shim.c" -o "$tmp/lapack_shim.o"
gcc -c -O2 -std=gnu99 -w -I"$src" "$here/../../experiments/native/vides_native.c" -o "$tmp/vides_native.o"
g++ -O2 -std=c++11 -w -I"$here" -I"$src" "$tmp/vides_gpu.cpp" "$here/cuda_emu.cpp" \
    "$tmp/vides_native.o" $OBJS "$tmp/lapack_shim.o" -lm -o "$out"
echo "built $out"
