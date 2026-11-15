// OpenMP implementation of HEO. Same algorithm and arithmetic as heo_seq.cpp (see heo_common.h), so results are
// bit-identical to the sequential reference for ANY thread count:
//  * every random number is a pure function of (seed, iteration, hare, dimension, stream) => independent of
//    which thread computes it and of the schedule;
//  * hares are independent within an iteration (each writes only its own candidate row) => no races, no atomics,
//    no critical sections;
//  * sorting uses a strict total order (fitness, index) => the same permutation whatever the thread count;
//  * each output rank k of the merge/selection is computed independently with a binary search (merge_select),
//    fused with the row gather => one parallel region instead of two.
// Per iteration: parallel update+evaluation | parallel sort of the N candidates | parallel selection+gather.
// schedule(static): every hare costs the same (D transcendental evaluations), so static chunks have the least
// scheduling overhead and keep each thread on contiguous rows (cache locality, little false sharing).
#define VERSION_NAME "omp"
#include "heo_common.h"
#include <algorithm>
#include <numeric>
#include <chrono>
#include <parallel/algorithm>   // __gnu_parallel::sort (libstdc++, needs -fopenmp)
#include <omp.h>

struct RowAcc { const real* p; HD real operator()(int i) const { return p[i]; } };

static int heo_backend_init(int threads) {
  if (threads > 0) omp_set_num_threads(threads);       // --threads wins over OMP_NUM_THREADS
  int n = omp_get_max_threads();
  fprintf(stderr, "[omp] threads = %d\n", n);
  return n;
}

struct Workspace {                                       // allocated once, reused by every run (see heo_seq.cpp)
  std::vector<real> pos, vel, pos2, vel2;
  std::vector<double> fit, fit2;
  std::vector<int> porder, cidx;
  void reserve(size_t N, size_t D) {
    size_t cnt = 2 * N * D;
    if (pos.size() < cnt) { pos.resize(cnt); vel.resize(cnt); pos2.resize(cnt); vel2.resize(cnt); }
    if (fit.size() < 2 * N) { fit.resize(2 * N); fit2.resize(2 * N); }
    if (porder.size() < N) { porder.resize(N); cidx.resize(N); }
  }
};

static double heo_run(const Params& p, uint64_t seed, RunInfo* info) {
  const int N = p.pop, D = p.dim, T = p.iters;
  static Workspace ws; ws.reserve((size_t)N, (size_t)D);
  real *pos = ws.pos.data(), *vel = ws.vel.data(), *pos2 = ws.pos2.data(), *vel2 = ws.vel2.data();
  double *fit = ws.fit.data(), *fit2 = ws.fit2.data();
  int *porder = ws.porder.data(), *cidx = ws.cidx.data();
  const uint64_t sseed = shift_seed(p.fid); const double half = 0.5 * (p.hi - p.lo);
  std::vector<double>* hist = info ? info->history : nullptr;
  if (hist) { hist->clear(); hist->reserve((size_t)T + 1); }

  auto c0 = std::chrono::steady_clock::now();
  #pragma omp parallel for schedule(static)
  for (int j = 0; j < N; j++) {                                     // initial population (A4)
    for (int i = 0; i < D; i++) {
      uint64_t e = (uint64_t)j * D + i;
      pos[(size_t)j * D + i] = (real)(p.lo + (p.hi - p.lo) * u01(seed, 0, e, 5));
      vel[(size_t)j * D + i] = (real)(p.lo + (p.hi - p.lo) * u01(seed, 0, e, 6));
    }
    ShiftAcc<RowAcc> s{RowAcc{&pos[(size_t)j * D]}, p.shift, sseed, half};
    fit[j] = bench_eval(p.fid, D, s, u01(seed, 0, j, 9));
  }
  std::iota(porder, porder + N, 0);
  __gnu_parallel::sort(porder, porder + N, [&](int a, int b) { return fit[a] < fit[b] || (fit[a] == fit[b] && a < b); });
  if (hist) hist->push_back(fit[porder[0]]);

  for (int t = 1; t <= T; t++) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < N; j++) {                                   // candidate + fitness, independent per hare
      for (int i = 0; i < D; i++) {
        uint64_t e = (uint64_t)j * D + i;
        real L = (real)(levy_step(seed, t, e, p.sigma) * p.levy_scale);
        real r = (real)u01(seed, t, e, 0);
        heo_step(pos[(size_t)j * D + i], vel[(size_t)j * D + i], t, T, p.vrc, p.damp, p.scale_mode, p.lo, p.hi, L, r,
                 &pos[(size_t)(N + j) * D + i], &vel[(size_t)(N + j) * D + i]);
      }
      ShiftAcc<RowAcc> s{RowAcc{&pos[(size_t)(N + j) * D]}, p.shift, sseed, half};
      fit[N + j] = bench_eval(p.fid, D, s, u01(seed, t, j, 9));
    }
    std::iota(cidx, cidx + N, 0);                                   // parallel sort of the N candidates (strict total order)
    __gnu_parallel::sort(cidx, cidx + N, [&](int a, int b) { return fit[N + a] < fit[N + b] || (fit[N + a] == fit[N + b] && a < b); });
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < N; k++) {                                   // survivor k: binary-search merge, then copy its row
      bool fromA; int idx;
      merge_select(k, N, [&](int q) { return fit[porder[q]]; }, [&](int q) { return fit[N + cidx[q]]; }, &fromA, &idx);
      int src = fromA ? porder[idx] : N + cidx[idx];
      memcpy(&pos2[(size_t)k * D], &pos[(size_t)src * D], D * sizeof(real));
      memcpy(&vel2[(size_t)k * D], &vel[(size_t)src * D], D * sizeof(real));
      fit2[k] = fit[src];
    }
    std::swap(pos, pos2); std::swap(vel, vel2); std::swap(fit, fit2);
    if (t == 1) std::iota(porder, porder + N, 0);                   // survivors are in rank order from now on
    if (hist) hist->push_back(fit[0]);
  }
  double best = fit[0];
  auto c1 = std::chrono::steady_clock::now();
  if (info) info->compute_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
  return best;
}
#include "heo_main.h"
