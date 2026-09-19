/* Implementation of the CUDA / cuBLAS emulation.  Not shipped. */
#include "cublas_v2.h"
#include "emu_linalg.h"
#include <cstdlib>
#include <cstring>
#include <map>

vides_dim3 blockIdx = {0,0,0}, blockDim = {1,1,1},
           threadIdx = {0,0,0}, gridDim = {1,1,1};

/* ---- dynamic shared memory ---------------------------------------- */
static double *g_shared = NULL;
static size_t  g_shared_bytes = 0;

double *vides_emu_shared(void) { return g_shared; }

void vides_emu_set_shmem(size_t bytes)
{
  if (bytes > g_shared_bytes) {
    free(g_shared);
    g_shared = (double *)calloc(bytes ? bytes : 8, 1);
    g_shared_bytes = bytes;
  }
}

/* ---- allocation tracking ------------------------------------------ */
static std::map<void *, size_t> g_allocs;
static size_t g_live = 0, g_peak = 0;

size_t vides_emu_live_bytes(void) { return g_live; }
size_t vides_emu_peak_bytes(void) { return g_peak; }
int    vides_emu_live_allocs(void) { return (int)g_allocs.size(); }

cudaError_t cudaMalloc(void **p, size_t n)
{
  if (n == 0) n = 1;
  void *q = malloc(n);
  if (!q) { *p = NULL; return cudaErrorUnknown; }
  /* Poison, so reads of uninitialised device memory misbehave visibly
     rather than happening to be zero. */
  memset(q, 0xA5, n);
  g_allocs[q] = n;
  g_live += n;
  if (g_live > g_peak) g_peak = g_live;
  *p = q;
  return cudaSuccess;
}

cudaError_t cudaFree(void *p)
{
  if (!p) return cudaSuccess;
  std::map<void *, size_t>::iterator it = g_allocs.find(p);
  if (it == g_allocs.end()) {
    fprintf(stderr, "[emu] cudaFree of untracked pointer %p\n", p);
    return cudaErrorUnknown;
  }
  g_live -= it->second;
  g_allocs.erase(it);
  free(p);
  return cudaSuccess;
}

cudaError_t cudaMemset(void *p, int v, size_t n)
{ memset(p, v, n); return cudaSuccess; }

cudaError_t cudaMemcpy(void *d, const void *s, size_t n, cudaMemcpyKind)
{ memmove(d, s, n); return cudaSuccess; }

cudaError_t cudaMemcpy2D(void *d, size_t dpitch, const void *s, size_t spitch,
                         size_t width, size_t height, cudaMemcpyKind)
{
  for (size_t r = 0; r < height; r++)
    memmove((char *)d + r*dpitch, (const char *)s + r*spitch, width);
  return cudaSuccess;
}

cudaError_t cudaGetDeviceCount(int *n) { *n = 1; return cudaSuccess; }

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *p, int)
{
  memset(p, 0, sizeof(*p));
  snprintf(p->name, sizeof p->name, "Emulated device (CPU)");
  p->major = 7; p->minor = 5;
  p->totalGlobalMem = (size_t)8 << 30;
  return cudaSuccess;
}

cudaError_t cudaMemGetInfo(size_t *freeb, size_t *totalb)
{ *totalb = (size_t)8 << 30; *freeb = (size_t)4 << 30; return cudaSuccess; }

cudaError_t cudaDeviceSynchronize(void) { return cudaSuccess; }
cudaError_t cudaGetLastError(void) { return cudaSuccess; }
const char *cudaGetErrorString(cudaError_t) { return "emulated"; }

/* ---- cuBLAS -------------------------------------------------------- */
cublasStatus_t cublasCreate(cublasHandle_t *h)
{ *h = (cublasHandle_t)1; return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasDestroy(cublasHandle_t) { return CUBLAS_STATUS_SUCCESS; }

cublasStatus_t cublasZgemmStridedBatched(
    cublasHandle_t, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const cuDoubleComplex *alpha,
    const cuDoubleComplex *A, int lda, long long strideA,
    const cuDoubleComplex *B, int ldb, long long strideB,
    const cuDoubleComplex *beta,
    cuDoubleComplex *C, int ldc, long long strideC, int batchCount)
{
  if (transa == CUBLAS_OP_T || transb == CUBLAS_OP_T)
    return CUBLAS_STATUS_FAILURE;     /* not used by vides_gpu.cu */
  for (int b = 0; b < batchCount; b++)
    emu_gemm(transa == CUBLAS_OP_C, transb == CUBLAS_OP_C, m, n, k,
             *(const emu_cplx *)alpha,
             (const emu_cplx *)A + b*strideA, lda,
             (const emu_cplx *)B + b*strideB, ldb,
             *(const emu_cplx *)beta,
             (emu_cplx *)C + b*strideC, ldc);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgetrfBatched(
    cublasHandle_t, int n, cuDoubleComplex *const A[], int lda,
    int *P, int *info, int batchSize)
{
  for (int b = 0; b < batchSize; b++)
    info[b] = emu_getrf((emu_cplx *)A[b], n, lda, P + (size_t)b*n);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgetriBatched(
    cublasHandle_t, int n, const cuDoubleComplex *const A[], int lda,
    const int *P, cuDoubleComplex *const C[], int ldc, int *info,
    int batchSize)
{
  for (int b = 0; b < batchSize; b++)
    info[b] = emu_getri((const emu_cplx *)A[b], n, lda, P + (size_t)b*n,
                        (emu_cplx *)C[b], ldc);
  return CUBLAS_STATUS_SUCCESS;
}
