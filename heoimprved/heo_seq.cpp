// Sequential reference implementation of Hare Escape Optimization (single thread).
// All algorithm details and documented paper ambiguities are in heo_common.h.
//
// Per iteration:  (1) candidate + fitness for every hare   (2) sort the N candidates by (fitness, index)
//                 (3) merge with the already-sorted parents, keep the best N (parents win ties)
//                 (4) gather the survivors.
// Steps 2-3 give exactly the same survivors, in the same order, as sorting all 2N elements with the tie
// rule "lower index first" (parents precede candidates), but cost O(N log N) instead of O(2N log 2N).
// Row j of the parent block is the hare that owns RNG elements j*D+i; after the first iteration the
// parents are stored in sorted order, so row index == rank (hence `porder` is the identity from then on).
#define VERSION_NAME "seq"
#include "heo_common.h"
#include <algorithm>
#include <numeric>
#include <chrono>

struct RowAcc { const real* p; HD real operator()(int i) const { return p[i]; } };
static int heo_backend_init(int) { return 1; }

// Workspace is allocated once and reused by every run (the untimed warm-up in heo_main.h sizes it),
// so the timed runs contain no heap allocation.
struct Workspace {
  std::vector<real> pos, vel, pos2, vel2;      // rows [0,N) parents, rows [N,2N) candidates; row = hare, D reals
  std::vector<double> fit, fit2;               // fit[0,N) parents, fit[N,2N) candidates ; fit2 = survivors
  std::vector<int> porder, cidx;               // parents in ascending order / candidates in ascending order
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
  auto eval = [&](const real* row, double noise) {
    ShiftAcc<RowAcc> s{RowAcc{row}, p.shift, sseed, half};
    return bench_eval(p.fid, D, s, noise); };
  std::vector<double>* hist = info ? info->history : nullptr;
  if (hist) { hist->clear(); hist->reserve((size_t)T + 1); }

  auto c0 = std::chrono::steady_clock::now();
  for (int j = 0; j < N; j++) {                                   // initial population (A4)
    for (int i = 0; i < D; i++) {
      uint64_t e = (uint64_t)j * D + i;
      pos[(size_t)j * D + i] = (real)(p.lo + (p.hi - p.lo) * u01(seed, 0, e, 5));
      vel[(size_t)j * D + i] = (real)(p.lo + (p.hi - p.lo) * u01(seed, 0, e, 6));
    }
    fit[j] = eval(&pos[(size_t)j * D], u01(seed, 0, j, 9));
  }
  std::iota(porder, porder + N, 0);                               // parents ordered by (fitness, index); rows stay put
  std::sort(porder, porder + N, [&](int a, int b) { return fit[a] < fit[b] || (fit[a] == fit[b] && a < b); });
  if (hist) hist->push_back(fit[porder[0]]);

  for (int t = 1; t <= T; t++) {
    for (int j = 0; j < N; j++) {                                 // candidate for each hare (Eq. 4)
      for (int i = 0; i < D; i++) {
        uint64_t e = (uint64_t)j * D + i;
        real L = (real)(levy_step(seed, t, e, p.sigma) * p.levy_scale);
        real r = (real)u01(seed, t, e, 0);
        heo_step(pos[(size_t)j * D + i], vel[(size_t)j * D + i], t, T, p.vrc, p.damp, p.scale_mode, p.lo, p.hi, L, r,
                 &pos[(size_t)(N + j) * D + i], &vel[(size_t)(N + j) * D + i]);
      }
      fit[N + j] = eval(&pos[(size_t)(N + j) * D], u01(seed, t, j, 9));
    }
    std::iota(cidx, cidx + N, 0);                                 // sort candidates by (fitness, index)
    std::sort(cidx, cidx + N, [&](int a, int b) { return fit[N + a] < fit[N + b] || (fit[N + a] == fit[N + b] && a < b); });
    int a = 0, b = 0;                                             // merge: keep best N, parents win ties (A6)
    for (int k = 0; k < N; k++) {
      bool fromA = (b >= N) || (a < N && fit[porder[a]] <= fit[N + cidx[b]]);
      int src = fromA ? porder[a++] : N + cidx[b++];
      memcpy(&pos2[(size_t)k * D], &pos[(size_t)src * D], D * sizeof(real));
      memcpy(&vel2[(size_t)k * D], &vel[(size_t)src * D], D * sizeof(real));
      fit2[k] = fit[src];
    }
    std::swap(pos, pos2); std::swap(vel, vel2); std::swap(fit, fit2);
    std::iota(porder, porder + N, 0);                             // survivors are stored in rank order
    if (hist) hist->push_back(fit[0]);
  }
  double best = fit[0];                                           // elitism: fit[0] is the best-so-far (never worse than any earlier best)
  auto c1 = std::chrono::steady_clock::now();
  if (info) info->compute_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
  // The swaps above may leave ws.pos/ws.pos2 exchanged, which is fine: both are scratch buffers of equal size.
  return best;
}
#include "heo_main.h"
