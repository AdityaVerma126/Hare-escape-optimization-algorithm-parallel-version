// heo_common.h - shared by the seq / OpenMP / CUDA builds.
// RNG, Levy flight, HEO update rule, selection helper and benchmark functions live here so
// all three versions execute the same math (counter-based RNG => results are reproducible and
// comparable across back-ends).
//
// ===========================================================================================
// PAPER-TO-CODE MAP (Alsamee & Ramezani, Sci. Rep. 15:26405, 2025) AND DOCUMENTED AMBIGUITIES
// ===========================================================================================
// Eq.(1)  D = L (X + V) r               -> the term  L*(x+v)*r  in heo_step(), per component.
//                                          L: fresh Levy step per (iteration, hare, dimension);
//                                          r ~ U(0,1) fresh per (iteration, hare, dimension).
// Eq.(4)  X' = X + (Maxit/t) V + L (X+V) r      -> heo_step(); V is the velocity *before* damping.
// Eq.(2)  V_{t+1} = (Maxit/t) V_t       -> NOT used literally (see A1).
// Pseudocode: initialise, then repeat { candidate for every hare by Eq.(4); merge old + new;
//             sort; keep the first N; first = best }       -> elitist selection (see A6).
//
// A1 Velocity update. Eq.(2) taken literally multiplies |V| by Maxit/t >= 1 every step, so V would
//    grow without bound, which contradicts "distance reduction". The experiments section states
//    "velocity reduction coefficient and damping are 1 and 0.95". Choice: V' = damp * V with
//    damp = 0.95; the reduction coefficient (vrc = 1) multiplies the position coefficient below.
// A2 Position coefficient. Eq.(3)/(4) use Maxit/t. Taken literally it equals T at t=1 and 1 at t=T:
//    it never drops below 1 and is huge early on (positions then saturate at the bounds). Both
//    readings are implemented and selected at run time with  --scale literal|decay :
//      decay   (DEFAULT, kept from the earlier version of this notebook): coef = vrc*(T-t+1)/T,
//              decreasing 1 -> 1/T, matching the text's "distance between hare and wolf is
//              gradually reduced";
//      literal : coef = vrc*T/t  (the equation exactly as printed).
//    The default is a documented interpretation, NOT the paper's printed equation.
// A3 Levy flight. The paper gives no algorithm or parameters. Choice: Mantegna's algorithm,
//    beta = 1.5, step = u / |v|^(1/beta), u ~ N(0, sigma_u^2), v ~ N(0,1), unscaled
//    (--levy-scale multiplies it). |v| is floored at 1e-12 to keep the step finite.
// A4 Initialisation. The pseudocode says "initialise random position for wolf V" (one wolf), yet
//    Eq.(4) uses a velocity V_{j,i} per hare. Choice: every hare gets its own position AND velocity,
//    both uniform in [lo, hi]^D from independent RNG streams.
// A5 Boundaries. The paper does not mention bound handling. Choice: clamp the new POSITION to
//    [lo, hi] (saturation). Velocity is not clamped (it only shrinks, by damping).
// A6 Selection. Parents and candidates are merged, sorted ascending by fitness and the best N are
//    kept. Ties: a parent beats a candidate; among parents / among candidates the lower index wins.
//    Because parents are always kept when better, the best fitness is monotonically non-increasing.
// A7 Iteration index. t runs 1..T (t = 0 would divide by zero in Maxit/t); Maxit = T = iters.
// A8 Benchmark typos in the paper (standard forms are used): Schwefel F18 lists fmin as
//    -418.9829 x 5 for d = 30 (here the function is offset so fmin = 0); Easom lacks a square on the
//    second exponent term; Beale's third term prints x1*x3^3; Schaffer N.2 is printed with a
//    non-standard layout. CEC 2015 F1-F15 need official shift/rotation data and are not implemented.
// A9 Noisy objective. F17 (Quartic) adds U[0,1) noise once per evaluation. A candidate's noisy value
//    is stored and elitism keeps it, so the reported best is the stored (noisy) fitness.
//
// FLOATING-POINT NOTE. Seq and OpenMP execute identical arithmetic (build with -ffp-contract=off) and
// agree bit for bit. CUDA uses its own sin/cos/log/exp/pow, which can differ from the host libm by
// ulps, so CUDA can differ at rounding level (and, rarely, through a changed selection order).
// CUB's radix sort orders -0.0 before +0.0 while '<' treats them as equal. Easom hits exactly +-0 on its
// plateau, so the CUDA kernels store fitness + 0.0 (maps -0.0 to +0.0) to keep the selection order identical.
#pragma once
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

#ifdef USE_FLOAT
typedef float real;          // storage type of positions/velocities (fitness stays double)
#else
typedef double real;
#endif

#define HEO_PI 3.14159265358979323846

// ---------------------------------------------------------------- RNG
// Stateless counter-based RNG (splitmix64 finaliser). u01(seed, a, b, c) -> uniform in the OPEN
// interval (0,1): never exactly 0 or 1, so log() below is always finite. Same value on CPU and GPU.
HD inline uint64_t mix64(uint64_t z) {
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
HD inline double u01(uint64_t seed, uint64_t a, uint64_t b, uint64_t c) {
  uint64_t h = mix64(seed ^ mix64(a ^ mix64(b ^ mix64(c))));
  return ((double)(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);     // (k + 0.5) / 2^53
}
// streams: 0=r  1..4=Levy  5=init x  6=init v  7=shift  9=quartic noise

// Levy flight step, Mantegna's algorithm, beta = 1.5 (A3)
HD inline double levy_step(uint64_t seed, uint64_t t, uint64_t e, double sigma) {
  double u1 = u01(seed, t, e, 1), u2 = u01(seed, t, e, 2);
  double u3 = u01(seed, t, e, 3), u4 = u01(seed, t, e, 4);
  double nu = sqrt(-2.0 * log(u1)) * cos(2.0 * HEO_PI * u2) * sigma;   // u ~ N(0, sigma^2)  (Box-Muller)
  double nv = sqrt(-2.0 * log(u3)) * cos(2.0 * HEO_PI * u4);           // v ~ N(0, 1)
  double an = fabs(nv); if (an < 1e-12) an = 1e-12;
  return nu / pow(an, 1.0 / 1.5);
}
static inline double levy_sigma_host(double beta = 1.5) {   // sigma_u(beta) ~ 0.69657 for beta = 1.5
  return pow(tgamma(1 + beta) * sin(HEO_PI * beta / 2) /
             (tgamma((1 + beta) / 2) * beta * pow(2.0, (beta - 1) / 2)), 1 / beta);
}

// ---------------------------------------------------------------- HEO update (Eq. 4)
// scale_mode 0 = "decay"  : coef = vrc*(T-t+1)/T   (default; see A2)
// scale_mode 1 = "literal": coef = vrc*Maxit/t     (paper equation as printed)
// X' = clamp( X + coef*V + L*(X+V)*r , lo, hi )   and   V' = damp*V          (A1, A5)
HD inline void heo_step(real x, real v, int t, int T, double vrc, double damp, int scale_mode,
                        double lo, double hi, real L, real r, real* xn, real* vn) {
  real coef = (scale_mode == 1) ? (real)(vrc * (double)T / (double)t)
                                : (real)(vrc * (double)(T - t + 1) / (double)T);
  real xx = x + coef * v + L * (x + v) * r;
  if (xx < (real)lo) xx = (real)lo;
  if (xx > (real)hi) xx = (real)hi;
  *xn = xx;
  *vn = (real)(damp * (double)v);
}

// ---------------------------------------------------------------- elitist selection helper (A6)
// A = parents in ascending order, B = candidates in ascending order, both of length n; A wins ties.
// merge_corank(k): how many of the first k merged elements come from A (binary search, O(log n)).
// merge_select(k): the k-th (0-based) merged element, k < n:  *fromA tells which list, *pos its index.
// Every output rank is independent, so OpenMP threads / CUDA threads can each take one rank.
template <class FA, class FB>
HD inline int merge_corank(int k, int n, FA a, FB b) {
  int lo = k > n ? k - n : 0, hi = k < n ? k : n;
  while (lo < hi) {
    int mid = lo + ((hi - lo) >> 1);
    if (a(mid) <= b(k - 1 - mid)) lo = mid + 1; else hi = mid;
  }
  return lo;
}
template <class FA, class FB>
HD inline void merge_select(int k, int n, FA a, FB b, bool* fromA, int* pos) {
  int i = merge_corank(k, n, a, b), j = k - i;           // for k < n both i < n and j < n
  if (a(i) <= b(j)) { *fromA = true; *pos = i; } else { *fromA = false; *pos = j; }
}

// ---------------------------------------------------------------- benchmarks
struct FuncInfo { const char* name; double lo, hi, fmin; int fixed_dim; int shiftable; };
static const FuncInfo FUNCS[] = {
  {"F16_StepSphere",  -100,  100, 0,          0, 1},
  {"F17_Quartic",     -1.28, 1.28, 0,         0, 1},   // + U[0,1) noise (A9)
  {"F18_Schwefel",    -500,  500, 0,          0, 0},   // offset so fmin = 0 (A8)
  {"F19_Rastrigin",   -5.12, 5.12, 0,         0, 1},
  {"Ackley",          -32,   32,  0,          0, 1},
  {"Griewank",        -600,  600, 0,          0, 1},
  {"Rosenbrock",      -30,   30,  0,          0, 0},
  {"F24_Bohachevsky", -100,  100, 0,          2, 0},
  {"F25_Booth",       -10,   10,  0,          2, 0},
  {"F26_Matyas",      -10,   10,  0,          2, 0},
  {"F27_Easom",       -100,  100, -1,         2, 0},
  {"F28_Beale",       -4.5,  4.5, 0,          2, 0},
  {"F29_SchafferN2",  -100,  100, 0,          2, 0},
  {"F30_SixHumpCamel",-2,    2,   -1.0316285, 2, 0},
};
static const int NFUNCS = (int)(sizeof(FUNCS) / sizeof(FUNCS[0]));

HD inline uint64_t shift_seed(int fid) { return 0xC0FFEEULL + (uint64_t)fid; }

// Wraps a coordinate accessor and (optionally) applies a fixed shift o_i
// (|o_i| <= 40% of half-range) so the optimum is NOT at the origin.
template <class A> struct ShiftAcc {
  A a; int on; uint64_t sseed; double half;
  HD double operator()(int i) const {
    double z = (double)a(i);
    if (on) z -= 0.4 * (2.0 * u01(sseed, 0, (uint64_t)i, 7) - 1.0) * half;
    return z;
  }
};

template <class A>
HD inline double bench_eval(int id, int D, A z, double noise) {
  double s = 0.0;
  switch (id) {
    case 0: for (int i = 0; i < D; i++) { double f = floor(z(i) + 0.5); s += f * f; } return s;
    case 1: for (int i = 0; i < D; i++) { double v = z(i); s += (i + 1) * v * v * v * v; } return s + noise;
    case 2: s = 418.9828872724337 * D;
            for (int i = 0; i < D; i++) { double v = z(i); s -= v * sin(sqrt(fabs(v))); } return s;
    case 3: for (int i = 0; i < D; i++) { double v = z(i); s += v * v - 10.0 * cos(2 * HEO_PI * v) + 10.0; } return s;
    case 4: { double s1 = 0, s2 = 0;
            for (int i = 0; i < D; i++) { double v = z(i); s1 += v * v; s2 += cos(2 * HEO_PI * v); }
            return -20.0 * exp(-0.2 * sqrt(s1 / D)) - exp(s2 / D) + 20.0 + 2.718281828459045; }
    case 5: { double p = 1.0;
            for (int i = 0; i < D; i++) { double v = z(i); s += v * v; p *= cos(v / sqrt(i + 1.0)); }
            return 1.0 + s / 4000.0 - p; }
    case 6: for (int i = 0; i < D - 1; i++) { double a = z(i), b = z(i + 1);
              s += 100.0 * (b - a * a) * (b - a * a) + (a - 1.0) * (a - 1.0); } return s;
    case 7: { double x = z(0), y = z(1);
            return x * x + 2 * y * y - 0.3 * cos(3 * HEO_PI * x) - 0.4 * cos(4 * HEO_PI * y) + 0.7; }
    case 8: { double x = z(0), y = z(1); double a = x + 2 * y - 7, b = 2 * x + y - 5; return a * a + b * b; }
    case 9: { double x = z(0), y = z(1); return 0.26 * (x * x + y * y) - 0.48 * x * y; }
    case 10:{ double x = z(0), y = z(1);
            return -cos(x) * cos(y) * exp(-((x - HEO_PI) * (x - HEO_PI) + (y - HEO_PI) * (y - HEO_PI))); }
    case 11:{ double x = z(0), y = z(1);
            double a = 1.5 - x + x * y, b = 2.25 - x + x * y * y, c = 2.625 - x + x * y * y * y;
            return a * a + b * b + c * c; }
    case 12:{ double x = z(0), y = z(1); double r2 = x * x + y * y; double sn = sin(sqrt(r2));
            double d = 1.0 + 0.001 * r2; return 0.5 + (sn * sn - 0.5) / (d * d); }
    case 13:{ double x = z(0), y = z(1);
            return 4 * x * x - 2.1 * x * x * x * x + x * x * x * x * x * x / 3.0 + x * y + (-4 + 4 * y * y) * y * y; }
  }
  return 0.0;
}

// ---------------------------------------------------------------- params / run info
struct Params {
  int fid = 0, dim = 30, pop = 30, iters = 500, runs = 10, shift = 0;
  uint64_t seed = 42;
  double vrc = 1.0, damp = 0.95;       // paper: velocity reduction coeff = 1, damping = 0.95
  double levy_scale = 1.0;             // paper uses raw L (scale 1); tunable via --levy-scale
  int scale_mode = 0;                  // 0 = decay (default), 1 = literal Maxit/t   (A2)
  int block = 256;                     // CUDA threads per block (ignored by CPU back-ends)
  double lo = 0, hi = 0, sigma = 0;
};

// Optional extra output of heo_run: compute-only time and best-so-far history.
struct RunInfo {
  double compute_ms = -1.0;                 // algorithm loop only (CUDA events on GPU, clock on CPU)
  std::vector<double>* history = nullptr;   // if set: history[0] = best after init, history[t] = after iteration t
};

// Returns an empty string if p is usable, otherwise a message describing the problem.
static inline std::string heo_validate(const Params& p) {
  if (p.fid < 0 || p.fid >= NFUNCS) return "function id out of range";
  if (p.dim < 1) return "dim must be >= 1";
  if (p.pop < 1) return "pop must be >= 1";
  if (p.iters < 1) return "iters must be >= 1";
  if (p.runs < 1) return "runs must be >= 1";
  if (p.block < 32 || p.block > 1024 || (p.block % 32)) return "block must be a multiple of 32 in [32,1024]";
  if (!(p.damp >= 0.0) || !(p.vrc >= 0.0) || !(p.levy_scale >= 0.0)) return "damp, vrc and levy-scale must be >= 0";
  if ((double)p.pop * 2.0 * (double)p.dim > 4.0e9) return "pop*dim too large (2*pop*dim must be <= 4e9 elements)";
  return "";
}
