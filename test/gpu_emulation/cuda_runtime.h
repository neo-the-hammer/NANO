/* Executable emulation of the CUDA runtime, for running vides_gpu.cu on a
 * machine with no GPU.  Device memory is host memory; kernel launches are
 * replayed serially over the grid.
 *
 * Fidelity limits, stated plainly:
 *   - threads in a block run one after another, so a kernel using
 *     __syncthreads() with more than one thread per block would be wrong.
 *     Only k_trace_prod does, and the test launches it with one thread per
 *     block, where its reduction degenerates correctly; the reduction tree
 *     itself is checked separately.
 *   - nothing here models coalescing, races, or launch configuration
 *     limits.  This checks algorithm and indexing, not performance or
 *     device-side legality.
 *
 * Not part of the shipped library.
 */
#ifndef VIDES_EMU_CUDA_RUNTIME_H
#define VIDES_EMU_CUDA_RUNTIME_H

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>

#define __global__
#define __device__
#define __host__
#define __shared__
#define __restrict__

struct vides_dim3 { unsigned int x, y, z; };
extern vides_dim3 blockIdx, blockDim, threadIdx, gridDim;

/* One block runs at a time, so a single dynamic-shared-memory buffer is
   enough.  emu_shared() is what the transformed `extern __shared__`
   declaration in vides_gpu.cu resolves to. */
double *vides_emu_shared(void);
void    vides_emu_set_shmem(size_t bytes);

inline void __syncthreads(void) {}
inline int  atomicExch(int *addr, int val) { int o = *addr; *addr = val; return o; }

/* Replay a launch serially over the whole grid. */
#define VIDES_EMU_LAUNCH(G, B, S, CALL)                                  \
  do {                                                                   \
    unsigned gg_ = (unsigned)(G), bb_ = (unsigned)(B);                   \
    vides_emu_set_shmem((size_t)(S));                                    \
    gridDim.x = gg_; blockDim.x = bb_;                                   \
    for (unsigned bi_ = 0; bi_ < gg_; ++bi_) {                           \
      blockIdx.x = bi_;                                                  \
      for (unsigned ti_ = 0; ti_ < bb_; ++ti_) {                         \
        threadIdx.x = ti_;                                               \
        CALL;                                                            \
      }                                                                  \
    }                                                                    \
  } while (0)

struct double2 { double x, y; };
typedef double2 cuDoubleComplex;
inline cuDoubleComplex make_cuDoubleComplex(double r, double i)
{ cuDoubleComplex z; z.x = r; z.y = i; return z; }

typedef enum { cudaSuccess = 0, cudaErrorUnknown = 1 } cudaError_t;
typedef enum {
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3
} cudaMemcpyKind;

struct cudaDeviceProp { char name[256]; int major, minor; size_t totalGlobalMem; };

/* Allocation is tracked so the test can report peak usage and catch a
   leak in the driver's unwind paths. */
size_t vides_emu_live_bytes(void);
size_t vides_emu_peak_bytes(void);
int    vides_emu_live_allocs(void);

cudaError_t cudaMalloc(void **p, size_t n);
cudaError_t cudaFree(void *p);
cudaError_t cudaMemset(void *p, int v, size_t n);
cudaError_t cudaMemcpy(void *d, const void *s, size_t n, cudaMemcpyKind k);
cudaError_t cudaMemcpy2D(void *d, size_t dpitch, const void *s, size_t spitch,
                         size_t width, size_t height, cudaMemcpyKind k);
cudaError_t cudaGetDeviceCount(int *n);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *p, int dev);
cudaError_t cudaMemGetInfo(size_t *freeb, size_t *totalb);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaGetLastError(void);
const char *cudaGetErrorString(cudaError_t e);

#endif
