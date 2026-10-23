// CUDA implementation. One thread per (hare, dimension) for update, one thread per hare for
// fitness evaluation, CUB radix sort (stable) for the merge/select step.
// Layout is dimension-major: element (hare j, dim i) at [i*M + j], M = 2N  => coalesced access.
// Columns [0,N) = parents, [N,2N) = candidates.
#define VERSION_NAME "cuda"
#include "heo_common.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <utility>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

struct ColAcc { const real* p; int stride; HD real operator()(int i) const { return p[(size_t)i * stride]; } };

__global__ void k_init(real* pos, real* vel, int N, int D, int M, uint64_t seed, double lo, double hi) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (tid >= (size_t)N * D) return;
  int j = (int)(tid % N), i = (int)(tid / N); uint64_t e = (uint64_t)j * D + i;
  pos[(size_t)i * M + j] = (real)(lo + (hi - lo) * u01(seed, 0, e, 5));
  vel[(size_t)i * M + j] = (real)(lo + (hi - lo) * u01(seed, 0, e, 6));
}
__global__ void k_update(real* pos, real* vel, int N, int D, int M, uint64_t seed, int t, int T,
                         double vrc, double damp, double lo, double hi, double sigma, double lscale) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (tid >= (size_t)N * D) return;
  int j = (int)(tid % N), i = (int)(tid / N); uint64_t e = (uint64_t)j * D + i;
  size_t a = (size_t)i * M + j, b = (size_t)i * M + N + j;
  real L = (real)(levy_step(seed, t, e, sigma) * lscale);
  real r = (real)u01(seed, t, e, 0);
  heo_step(pos[a], vel[a], t, T, vrc, damp, lo, hi, L, r, &pos[b], &vel[b]);
}
__global__ void k_eval(const real* pos, double* fit, int off, int N, int D, int M, int fid, int shift,
                       double lo, double hi, uint64_t seed, int t) {
  int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= N) return;
  ColAcc a{pos + off + j, M};
  ShiftAcc<ColAcc> s{a, shift, shift_seed(fid), 0.5 * (hi - lo)};
  fit[off + j] = bench_eval(fid, D, s, u01(seed, t, j, 9));
}
__global__ void k_iota(int* idx, int M) { int k = blockIdx.x * blockDim.x + threadIdx.x; if (k < M) idx[k] = k; }
__global__ void k_gather(const real* pos, const real* vel, real* pos2, real* vel2, const int* idx, int N, int D, int M) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (tid >= (size_t)N * D) return;
  int j = (int)(tid % N), i = (int)(tid / N); int s = idx[j];
  pos2[(size_t)i * M + j] = pos[(size_t)i * M + s];
  vel2[(size_t)i * M + j] = vel[(size_t)i * M + s];
}

static void heo_backend_init() {
  CK(cudaFree(0));
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  fprintf(stderr, "[cuda] device = %s (%d SMs)\n", pr.name, pr.multiProcessorCount);
}

static double heo_run(const Params& p, uint64_t seed) {
  const int N = p.pop, D = p.dim, M = 2 * N, T = p.iters;
  const size_t cnt = (size_t)M * D;
  real *pos, *vel, *pos2, *vel2; double *fit, *fit2; int *idx_in, *idx_out;
  CK(cudaMalloc(&pos, cnt * sizeof(real)));  CK(cudaMalloc(&vel, cnt * sizeof(real)));
  CK(cudaMalloc(&pos2, cnt * sizeof(real))); CK(cudaMalloc(&vel2, cnt * sizeof(real)));
  CK(cudaMalloc(&fit, M * sizeof(double)));  CK(cudaMalloc(&fit2, M * sizeof(double)));
  CK(cudaMalloc(&idx_in, M * sizeof(int)));  CK(cudaMalloc(&idx_out, M * sizeof(int)));
  void* tmp = nullptr; size_t tmpb = 0;
  CK(cub::DeviceRadixSort::SortPairs(nullptr, tmpb, fit, fit2, idx_in, idx_out, M));
  CK(cudaMalloc(&tmp, tmpb));

  const int B = 256;
  const unsigned gND = (unsigned)(((size_t)N * D + B - 1) / B), gN = (N + B - 1) / B, gM = (M + B - 1) / B;
  k_init<<<gND, B>>>(pos, vel, N, D, M, seed, p.lo, p.hi);
  k_eval<<<gN, B>>>(pos, fit, 0, N, D, M, p.fid, p.shift, p.lo, p.hi, seed, 0);
  for (int t = 1; t <= T; t++) {
    k_update<<<gND, B>>>(pos, vel, N, D, M, seed, t, T, p.vrc, p.damp, p.lo, p.hi, p.sigma, p.levy_scale);
    k_eval<<<gN, B>>>(pos, fit, N, N, D, M, p.fid, p.shift, p.lo, p.hi, seed, t);
    k_iota<<<gM, B>>>(idx_in, M);
    CK(cub::DeviceRadixSort::SortPairs(tmp, tmpb, fit, fit2, idx_in, idx_out, M));   // stable
    k_gather<<<gND, B>>>(pos, vel, pos2, vel2, idx_out, N, D, M);
    std::swap(pos, pos2); std::swap(vel, vel2); std::swap(fit, fit2);
  }
  CK(cudaGetLastError());
  double best; CK(cudaMemcpy(&best, fit, sizeof(double), cudaMemcpyDeviceToHost));   // implicit sync
  cudaFree(pos); cudaFree(vel); cudaFree(pos2); cudaFree(vel2); cudaFree(fit); cudaFree(fit2);
  cudaFree(idx_in); cudaFree(idx_out); cudaFree(tmp);
  return best;
}
#include "heo_main.h"
