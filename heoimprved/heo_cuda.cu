// CUDA implementation of HEO. Same algorithm and RNG streams as heo_seq.cpp (details: heo_common.h).
//
// Data layout: dimension-major, element (hare j, dim i) at [i*M + j], M = 2N. Columns [0,N) = parents,
// [N,2N) = candidates. Consecutive threads handle consecutive hares j, so every position/velocity access in the
// update, evaluation and gather kernels is coalesced.
//
// Everything stays on the device for the whole run. The ONLY host<->device transfers are the final best fitness
// (8 bytes) and, if --history is used, the history array. Buffers are allocated once and reused by every run
// (the untimed warm-up in heo_main.h sizes them).
//
// Per iteration (all on the default stream):
//   k_update   one thread per (hare, dimension): candidate (Eq. 4), coalesced
//   evaluation one thread per hare (k_eval)  -- or --  one thread per (hare, dimension) (k_terms) followed by
//              k_reduce (see "ONE THREAD PER HARE vs SEVERAL" below)
//   CUB radix sort of the N candidate fitnesses (stable => ties keep ascending index)
//   k_select   one thread per survivor rank: binary-search merge of sorted parents and sorted candidates
//   k_gather   one thread per (hare, dimension): copy the survivors (coalesced writes)
// Compared with sorting all 2N elements, only N are sorted and the former k_iota launch is gone (the iota array is
// constant); the sort of the parents happens once, before iteration 1.
//
// ONE THREAD PER HARE vs SEVERAL (evaluation). Update and gather already use one thread per (hare, dimension).
// Evaluation sums D terms per hare. With one thread per hare (default, HEO_CUDA_EVAL=thread) only N threads run,
// which under-fills a GPU when N is small (a T4 holds ~40k resident threads). A block-wide reduction would use
// more threads but changes the floating-point summation order, breaking bit-level parity with the CPU versions.
// HEO_CUDA_EVAL=terms keeps the order: all N*D terms (the expensive sin/cos/exp calls) are computed in parallel by
// k_terms, then k_reduce adds them left to right exactly as bench_eval does (one thread per hare, coalesced
// reads). It applies to the scalable functions (ids 0-6); others always use k_eval. HEO_CUDA_EVAL=auto picks terms
// when N < 8192 and D >= 64. Which mode is faster depends on the GPU and the sizes: measure it (Section 7).
//
// Timing: compute_ms = CUDA events around the whole algorithm loop (first kernel -> last kernel). The caller's
// time_ms additionally includes host launch latency, the final synchronisation and the 8-byte copy back.
#define VERSION_NAME "cuda"
#include "heo_common.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <utility>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

struct ColAcc { const real* p; int stride; HD real operator()(int i) const { return p[(size_t)i * stride]; } };
struct ParentKey { const double* fit; const int* order; HD double operator()(int q) const { return fit[order[q]]; } };
struct SortedKey { const double* key; HD double operator()(int q) const { return key[q]; } };

__global__ void k_init(real* pos, real* vel, int N, int D, int M, uint64_t seed, double lo, double hi) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (tid >= (size_t)N * D) return;
  int j = (int)(tid % N), i = (int)(tid / N); uint64_t e = (uint64_t)j * D + i;
  pos[(size_t)i * M + j] = (real)(lo + (hi - lo) * u01(seed, 0, e, 5));
  vel[(size_t)i * M + j] = (real)(lo + (hi - lo) * u01(seed, 0, e, 6));
}
__global__ void k_update(real* pos, real* vel, int N, int D, int M, uint64_t seed, int t, int T,
                         double vrc, double damp, int scale_mode, double lo, double hi, double sigma, double lscale) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (tid >= (size_t)N * D) return;
  int j = (int)(tid % N), i = (int)(tid / N); uint64_t e = (uint64_t)j * D + i;
  size_t a = (size_t)i * M + j, b = (size_t)i * M + N + j;
  real L = (real)(levy_step(seed, t, e, sigma) * lscale);
  real r = (real)u01(seed, t, e, 0);
  heo_step(pos[a], vel[a], t, T, vrc, damp, scale_mode, lo, hi, L, r, &pos[b], &vel[b]);
}
// evaluation, one thread per hare: columns [off, off+N)
__global__ void k_eval(const real* pos, double* fit, int off, int N, int D, int M, int fid, int shift,
                       double lo, double hi, uint64_t seed, int t) {
  int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= N) return;
  ColAcc a{pos + off + j, M};
  ShiftAcc<ColAcc> s{a, shift, shift_seed(fid), 0.5 * (hi - lo)};
  fit[off + j] = bench_eval(fid, D, s, u01(seed, t, j, 9)) + 0.0;   // +0.0 turns -0.0 into +0.0 (see k_reduce)
}
// evaluation, several threads per hare: term kernel + order-preserving reduction (ids 0-6 only).
// Expressions are copied from bench_eval so each term, and the left-to-right sum, is bit-identical.
__global__ void k_terms(const real* pos, double* t1, double* t2, int off, int N, int D, int M, int fid, int shift,
                        double lo, double hi) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (tid >= (size_t)N * D) return;
  int j = (int)(tid % N), i = (int)(tid / N);
  ColAcc a{pos + off + j, M};
  ShiftAcc<ColAcc> z{a, shift, shift_seed(fid), 0.5 * (hi - lo)};
  double v = z(i), x1 = 0.0, x2 = 0.0;
  switch (fid) {
    case 0: { double f = floor(v + 0.5); x1 = f * f; break; }
    case 1: x1 = (i + 1) * v * v * v * v; break;
    case 2: x1 = v * sin(sqrt(fabs(v))); break;
    case 3: x1 = v * v - 10.0 * cos(2 * HEO_PI * v) + 10.0; break;
    case 4: x1 = v * v; x2 = cos(2 * HEO_PI * v); break;
    case 5: x1 = v * v; x2 = cos(v / sqrt(i + 1.0)); break;
    case 6: if (i < D - 1) { double aa = v, bb = z(i + 1);
              x1 = 100.0 * (bb - aa * aa) * (bb - aa * aa) + (aa - 1.0) * (aa - 1.0); } break;
  }
  t1[(size_t)i * N + j] = x1; t2[(size_t)i * N + j] = x2;
}
__global__ void k_reduce(const double* t1, const double* t2, double* fit, int off, int N, int D, int fid,
                         uint64_t seed, int t) {
  int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= N) return;
  double s = 0.0, s2 = 0.0, p = 1.0;
  if (fid == 2) s = 418.9828872724337 * D;
  int n = (fid == 6) ? D - 1 : D;
  for (int i = 0; i < n; i++) {
    double a = t1[(size_t)i * N + j];
    if (fid == 2) s -= a; else s += a;
    if (fid == 4) s2 += t2[(size_t)i * N + j];
    if (fid == 5) p *= t2[(size_t)i * N + j];
  }
  double f;
  switch (fid) {
    case 1:  f = s + u01(seed, t, j, 9); break;
    case 4:  f = -20.0 * exp(-0.2 * sqrt(s / D)) - exp(s2 / D) + 20.0 + 2.718281828459045; break;
    case 5:  f = 1.0 + s / 4000.0 - p; break;
    default: f = s; break;
  }
  // CUB's radix sort orders -0.0 before +0.0, while the CPU versions treat them as equal (tie -> lower index).
  // Easom underflows to exactly +-0 on its plateau, so normalise: x + 0.0 maps -0.0 to +0.0 and changes no other value.
  fit[off + j] = f + 0.0;
}
__global__ void k_iota(int* idx, int n) { int k = blockIdx.x * blockDim.x + threadIdx.x; if (k < n) idx[k] = k; }
// survivor k = k-th element of merge(parents, candidates), parents win ties (A6)
__global__ void k_select(const double* fit, double* fit2, const int* porder, const double* ckey, const int* cidx,
                         int* sel, int N) {
  int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= N) return;
  bool fromA; int idx;
  merge_select(k, N, ParentKey{fit, porder}, SortedKey{ckey}, &fromA, &idx);
  int src = fromA ? porder[idx] : N + cidx[idx];
  sel[k] = src; fit2[k] = fit[src];
}
__global__ void k_gather(const real* pos, const real* vel, real* pos2, real* vel2, const int* sel, int N, int D, int M) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (tid >= (size_t)N * D) return;
  int j = (int)(tid % N), i = (int)(tid / N); int s = sel[j];
  pos2[(size_t)i * M + j] = pos[(size_t)i * M + s];
  vel2[(size_t)i * M + j] = vel[(size_t)i * M + s];
}
__global__ void k_record(const double* fit, const int* order, double* hist, int t) {   // hist[t] = best so far
  hist[t] = fit[order ? order[0] : 0];
}

// ---------------------------------------------------------------- persistent device workspace
struct Workspace {
  size_t capN = 0, capD = 0, capTerm = 0, capHist = 0, tmpb = 0;
  real *pos = 0, *vel = 0, *pos2 = 0, *vel2 = 0;
  double *fit = 0, *fit2 = 0, *ckey = 0, *pkey = 0, *t1 = 0, *t2 = 0, *hist = 0;
  int *iota_ = 0, *porder = 0, *cidx = 0, *sel = 0;
  void* tmp = 0; cudaEvent_t e0, e1; bool ev = false;
  void free_main() {
    cudaFree(pos); cudaFree(vel); cudaFree(pos2); cudaFree(vel2); cudaFree(fit); cudaFree(fit2); cudaFree(ckey);
    cudaFree(pkey); cudaFree(iota_); cudaFree(porder); cudaFree(cidx); cudaFree(sel);
    pos = vel = pos2 = vel2 = 0; fit = fit2 = ckey = pkey = 0; iota_ = porder = cidx = sel = 0;
  }
  void reserve(int N, int D, bool terms, int Hn) {
    if (!ev) { CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); ev = true; }
    if ((size_t)N > capN || (size_t)D > capD) {
      free_main(); capN = N > (int)capN ? N : capN; capD = D > (int)capD ? D : capD;
      size_t cnt = 2 * capN * capD;
      CK(cudaMalloc(&pos, cnt * sizeof(real)));  CK(cudaMalloc(&vel, cnt * sizeof(real)));
      CK(cudaMalloc(&pos2, cnt * sizeof(real))); CK(cudaMalloc(&vel2, cnt * sizeof(real)));
      CK(cudaMalloc(&fit, 2 * capN * sizeof(double))); CK(cudaMalloc(&fit2, 2 * capN * sizeof(double)));
      CK(cudaMalloc(&ckey, capN * sizeof(double)));    CK(cudaMalloc(&pkey, capN * sizeof(double)));
      CK(cudaMalloc(&iota_, capN * sizeof(int)));      CK(cudaMalloc(&porder, capN * sizeof(int)));
      CK(cudaMalloc(&cidx, capN * sizeof(int)));       CK(cudaMalloc(&sel, capN * sizeof(int)));
    }
    size_t need = 0;                                            // CUB temp storage for sorting N (key,int) pairs
    CK(cub::DeviceRadixSort::SortPairs(nullptr, need, fit, ckey, iota_, cidx, N));
    if (need > tmpb) { cudaFree(tmp); tmp = 0; CK(cudaMalloc(&tmp, need)); tmpb = need; }
    if (terms && (size_t)N * D > capTerm) {
      cudaFree(t1); cudaFree(t2); capTerm = (size_t)N * D;
      CK(cudaMalloc(&t1, capTerm * sizeof(double))); CK(cudaMalloc(&t2, capTerm * sizeof(double)));
    }
    if (Hn > 0 && (size_t)Hn > capHist) { cudaFree(hist); capHist = Hn; CK(cudaMalloc(&hist, capHist * sizeof(double))); }
  }
};
static Workspace ws;

static const char* g_eval_env = "thread";
static int heo_backend_init(int) {
  CK(cudaFree(0));
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  const char* e = getenv("HEO_CUDA_EVAL"); g_eval_env = (e && *e) ? e : "thread";
  if (strcmp(g_eval_env, "thread") && strcmp(g_eval_env, "terms") && strcmp(g_eval_env, "auto")) {
    fprintf(stderr, "error: HEO_CUDA_EVAL must be thread, terms or auto (got '%s')\n", g_eval_env); exit(2); }
  fprintf(stderr, "[cuda] device = %s (%d SMs) | eval mode = %s\n", pr.name, pr.multiProcessorCount, g_eval_env);
  return 0;
}

static double heo_run(const Params& p, uint64_t seed, RunInfo* info) {
  const int N = p.pop, D = p.dim, M = 2 * N, T = p.iters, B = p.block;
  std::vector<double>* hist = info ? info->history : nullptr;
  bool terms = false;
  if (p.fid <= 6 && strcmp(g_eval_env, "thread") != 0)
    terms = strcmp(g_eval_env, "terms") == 0 || (N < 8192 && D >= 64);
  ws.reserve(N, D, terms, hist ? T + 1 : 0);
  real *pos = ws.pos, *vel = ws.vel, *pos2 = ws.pos2, *vel2 = ws.vel2; double *fit = ws.fit, *fit2 = ws.fit2;
  const unsigned gND = (unsigned)(((size_t)N * D + B - 1) / B), gN = (unsigned)((N + B - 1) / B);

  auto evaluate = [&](int off, int t) {                              // fitness of columns [off, off+N)
    if (terms) {
      k_terms<<<gND, B>>>(pos, ws.t1, ws.t2, off, N, D, M, p.fid, p.shift, p.lo, p.hi);
      k_reduce<<<gN, B>>>(ws.t1, ws.t2, fit, off, N, D, p.fid, seed, t);
    } else {
      k_eval<<<gN, B>>>(pos, fit, off, N, D, M, p.fid, p.shift, p.lo, p.hi, seed, t);
    }
  };

  CK(cudaEventRecord(ws.e0));
  k_iota<<<gN, B>>>(ws.iota_, N);
  k_init<<<gND, B>>>(pos, vel, N, D, M, seed, p.lo, p.hi);
  evaluate(0, 0);
  CK(cub::DeviceRadixSort::SortPairs(ws.tmp, ws.tmpb, fit, ws.pkey, ws.iota_, ws.porder, N));   // parents by (fitness, index)
  if (hist) k_record<<<1, 1>>>(fit, ws.porder, ws.hist, 0);
  for (int t = 1; t <= T; t++) {
    k_update<<<gND, B>>>(pos, vel, N, D, M, seed, t, T, p.vrc, p.damp, p.scale_mode, p.lo, p.hi, p.sigma, p.levy_scale);
    evaluate(N, t);
    CK(cub::DeviceRadixSort::SortPairs(ws.tmp, ws.tmpb, fit + N, ws.ckey, ws.iota_, ws.cidx, N));  // stable: ties keep index order
    const int* po = (t == 1) ? ws.porder : ws.iota_;                  // survivors are in rank order after iteration 1
    k_select<<<gN, B>>>(fit, fit2, po, ws.ckey, ws.cidx, ws.sel, N);
    k_gather<<<gND, B>>>(pos, vel, pos2, vel2, ws.sel, N, D, M);
    std::swap(pos, pos2); std::swap(vel, vel2); std::swap(fit, fit2);
    if (hist) k_record<<<1, 1>>>(fit, nullptr, ws.hist, t);
    CK(cudaGetLastError());
  }
  CK(cudaEventRecord(ws.e1));
  CK(cudaEventSynchronize(ws.e1));
  if (info) { float ms = 0.f; CK(cudaEventElapsedTime(&ms, ws.e0, ws.e1)); info->compute_ms = (double)ms; }
  double best; CK(cudaMemcpy(&best, fit, sizeof(double), cudaMemcpyDeviceToHost));   // survivors sorted: fit[0] is the best
  if (hist) { hist->resize((size_t)T + 1); CK(cudaMemcpy(hist->data(), ws.hist, ((size_t)T + 1) * sizeof(double), cudaMemcpyDeviceToHost)); }
  return best;
}
#include "heo_main.h"
