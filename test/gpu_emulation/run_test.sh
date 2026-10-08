#!/bin/sh
# Execute vides_gpu.cu's driver on the CPU and check it against the stock
# NEGF path.  Needs no GPU, no CUDA toolkit and no LAPACK.
#
#   sh test/gpu_emulation/run_test.sh
#
# vides_gpu.cu is transformed so it can be compiled by g++:
#   - `kernel<<<g,b[,s]>>>(args)` becomes `vides_launch(kernel,g,b,s)(args)`,
#     a functor that replays the launch serially over the grid.  Only the
#     launch configuration is rewritten; the argument list is untouched.
#   - the dynamic-shared-memory declaration is pointed at the emulator's
#     buffer.
#   - k_trace_prod is launched with one thread per block, where its
#     reduction degenerates to a plain sum; the reduction tree itself is
#     checked separately in emu_main.cpp.
# The algorithm, indexing and cuBLAS argument order under test are the
# real ones.

set -e
here=$(cd "$(dirname "$0")" && pwd)
src=$here/../../src
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

cat > "$tmp/launch.h" <<'EOF'
#ifndef VIDES_EMU_LAUNCH_H
#define VIDES_EMU_LAUNCH_H
#include "cuda_runtime.h"
template <class F> struct EmuLauncher {
  F f; unsigned g, b; size_t s;
  template <class... A> void operator()(A... a) const {
    vides_emu_set_shmem(s);
    gridDim.x = g; blockDim.x = b;
    for (unsigned bi = 0; bi < g; ++bi) { blockIdx.x = bi;
      for (unsigned ti = 0; ti < b; ++ti) { threadIdx.x = ti; f(a...); } }
  }
};
template <class F>
EmuLauncher<F> vides_launch(F f, unsigned g, unsigned b, size_t s)
{ EmuLauncher<F> L; L.f = f; L.g = g; L.b = b; L.s = s; return L; }
#endif
EOF

perl -0777 -pe '
  s{(\w+)\s*<<<(.*?)>>>}{
    my ($name, $cfg) = ($1, $2);
    my @parts; my $depth = 0; my $cur = "";
    for my $ch (split //, $cfg) {
      if    ($ch eq "(") { $depth++; $cur .= $ch }
      elsif ($ch eq ")") { $depth--; $cur .= $ch }
      elsif ($ch eq "," && $depth == 0) { push @parts, $cur; $cur = "" }
      else  { $cur .= $ch }
    }
    push @parts, $cur;
    my $s = defined $parts[2] ? $parts[2] : "0";
    "vides_launch($name, $parts[0], $parts[1], $s)"
  }ge;
  s{extern\s+__shared__\s+double\s+sh\[\];}{double *sh = vides_emu_shared();};
  s{const int tpb_red = 128;}{const int tpb_red = 1;};
' "$src/vides_gpu.cu" > "$tmp/vides_gpu.cpp"

# The launch helper has to be visible before the first kernel launch.
sed -i 's|#include "vides_rgf_batch.h"|#include "vides_rgf_batch.h"\n#include "launch.h"|' "$tmp/vides_gpu.cpp"

CFILES="nrutil cmatrix cmatrixm cvectorm ctensor4 c3tensor ccvector \
        cfree_cmatrix cfree_cvectorm cfree_ctensor4 cfree_c3tensor cfree_ccvector \
        complass csum csub cmatsub cmatsum cmatdaga cmatmul cmatmul3 cmatinv cmatRe \
        cdabs cIm cmatmul_proc spectralfun spectralfunmode transmission VAVdaga \
        rgfblock rgfblock_Lake LDOS LDOS_Lake LDOSmode vides_rgf_batch \
        Gzerozero cmatnorm2diff selfGNR selfH_dec create_updiagGNR \
        create_lowdiagGNR create_beta1GNR create_beta2GNR create_beta2transpGNR"

OBJS=""
for f in $CFILES; do
  gcc -c -O1 -std=gnu89 -w -I"$src" -I"$here" "$src/$f.c" -o "$tmp/$f.o"
  OBJS="$OBJS $tmp/$f.o"
done
gcc -c -O1 -w -I"$here" "$here/lapack_shim.c" -o "$tmp/lapack_shim.o"

g++ -O1 -std=c++11 -w -I"$here" -I"$src" \
    "$tmp/vides_gpu.cpp" "$here/cuda_emu.cpp" "$here/emu_main.cpp" \
    $OBJS "$tmp/lapack_shim.o" -lm -o "$tmp/emu_test"

"$tmp/emu_test"
