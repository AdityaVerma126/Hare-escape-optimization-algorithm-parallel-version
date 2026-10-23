// Sequential reference implementation of Hare Escape Optimization.
#define VERSION_NAME "seq"
#include "heo_common.h"
#include <algorithm>
#include <numeric>

struct RowAcc { const real* p; HD real operator()(int i) const { return p[i]; } };
static void heo_backend_init() {}

static double heo_run(const Params& p, uint64_t seed) {
  const int N = p.pop, D = p.dim, M = 2 * N, T = p.iters;
  const size_t cnt = (size_t)M * D;
  std::vector<real> pos(cnt), vel(cnt), pos2(cnt), vel2(cnt);   // row j = hare j (AoS)
  std::vector<double> fit(M), fit2(M);
  std::vector<int> idx(M);
  const uint64_t sseed = shift_seed(p.fid); const double half = 0.5 * (p.hi - p.lo);
  auto eval = [&](const real* row, double noise) {
    ShiftAcc<RowAcc> s{RowAcc{row}, p.shift, sseed, half};
    return bench_eval(p.fid, D, s, noise); };

  for (int j = 0; j < N; j++) {
    for (int i = 0; i < D; i++) {
      uint64_t e = (uint64_t)j * D + i;
      pos[(size_t)j * D + i] = (real)(p.lo + (p.hi - p.lo) * u01(seed, 0, e, 5));
      vel[(size_t)j * D + i] = (real)(p.lo + (p.hi - p.lo) * u01(seed, 0, e, 6));
    }
    fit[j] = eval(&pos[(size_t)j * D], u01(seed, 0, j, 9));
  }
  for (int t = 1; t <= T; t++) {
    for (int j = 0; j < N; j++) {                       // candidate for each hare
      for (int i = 0; i < D; i++) {
        uint64_t e = (uint64_t)j * D + i;
        real L = (real)(levy_step(seed, t, e, p.sigma) * p.levy_scale);
        real r = (real)u01(seed, t, e, 0);
        heo_step(pos[(size_t)j * D + i], vel[(size_t)j * D + i], t, T, p.vrc, p.damp, p.lo, p.hi, L, r,
                 &pos[(size_t)(N + j) * D + i], &vel[(size_t)(N + j) * D + i]);
      }
      fit[N + j] = eval(&pos[(size_t)(N + j) * D], u01(seed, t, j, 9));
    }
    std::iota(idx.begin(), idx.end(), 0);               // merge old+new, keep best N (ties -> lower index)
    std::partial_sort(idx.begin(), idx.begin() + N, idx.end(),
        [&](int a, int b) { return fit[a] < fit[b] || (fit[a] == fit[b] && a < b); });
    for (int k = 0; k < N; k++) {
      int s = idx[k];
      memcpy(&pos2[(size_t)k * D], &pos[(size_t)s * D], D * sizeof(real));
      memcpy(&vel2[(size_t)k * D], &vel[(size_t)s * D], D * sizeof(real));
      fit2[k] = fit[s];
    }
    pos.swap(pos2); vel.swap(vel2); fit.swap(fit2);
  }
  return fit[0];
}
#include "heo_main.h"
