// heo_common.h - shared by the seq / OpenMP / CUDA builds.
// RNG, Levy flight, HEO update rule and benchmark functions live here so all
// three versions execute *exactly* the same math (counter-based RNG => results
// are reproducible and comparable across back-ends).
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
typedef float real;          // storage type of positions/velocities
#else
typedef double real;
#endif

#define HEO_PI 3.14159265358979323846

// ---------------------------------------------------------------- RNG
// Stateless counter-based RNG (splitmix64). u01(seed, iter, element, stream)
// -> uniform in the OPEN interval (0,1). Same value on CPU and GPU.
HD inline uint64_t mix64(uint64_t z) {
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
HD inline double u01(uint64_t seed, uint64_t a, uint64_t b, uint64_t c) {
  uint64_t h = mix64(seed ^ mix64(a ^ mix64(b ^ mix64(c))));
  return ((double)(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
// streams: 0=r  1..4=Levy  5=init x  6=init v  7=shift  9=quartic noise

// Levy flight step, Mantegna's algorithm, beta = 1.5
HD inline double levy_step(uint64_t seed, uint64_t t, uint64_t e, double sigma) {
  double u1 = u01(seed, t, e, 1), u2 = u01(seed, t, e, 2);
  double u3 = u01(seed, t, e, 3), u4 = u01(seed, t, e, 4);
  double nu = sqrt(-2.0 * log(u1)) * cos(2.0 * HEO_PI * u2) * sigma;
  double nv = sqrt(-2.0 * log(u3)) * cos(2.0 * HEO_PI * u4);
  double an = fabs(nv); if (an < 1e-12) an = 1e-12;
  return nu / pow(an, 1.0 / 1.5);
}
static inline double levy_sigma_host(double beta = 1.5) {
  return pow(tgamma(1 + beta) * sin(HEO_PI * beta / 2) /
             (tgamma((1 + beta) / 2) * beta * pow(2.0, (beta - 1) / 2)), 1 / beta);
}

// ---------------------------------------------------------------- HEO update
// Paper Eq.(4):  X' = X + (Maxit/t) V + L (X+V) r ,   V' = damping * V
// Default coefficient: vrc*(T-t+1)/T  (decreasing 1 -> 0, i.e. "distance reduction").
// Compile with -DHEO_LITERAL_SCALE to use the literal Maxit/t from the paper.
HD inline void heo_step(real x, real v, int t, int T, double vrc, double damp,
                        double lo, double hi, real L, real r, real* xn, real* vn) {
#ifdef HEO_LITERAL_SCALE
  real coef = (real)(vrc * (double)T / (double)t);
#else
  real coef = (real)(vrc * (double)(T - t + 1) / (double)T);
#endif
  real xx = x + coef * v + L * (x + v) * r;
  if (xx < (real)lo) xx = (real)lo;
  if (xx > (real)hi) xx = (real)hi;
  *xn = xx;
  *vn = (real)(damp * (double)v);
}

// ---------------------------------------------------------------- benchmarks
struct FuncInfo { const char* name; double lo, hi, fmin; int fixed_dim; int shiftable; };
static const FuncInfo FUNCS[] = {
  {"F16_StepSphere",  -100,  100, 0,          0, 1},
  {"F17_Quartic",     -1.28, 1.28, 0,         0, 1},   // + U[0,1) noise
  {"F18_Schwefel",    -500,  500, 0,          0, 0},   // offset so fmin = 0
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

// ---------------------------------------------------------------- params
struct Params {
  int fid = 0, dim = 30, pop = 30, iters = 500, runs = 10, shift = 0;
  uint64_t seed = 42;
  double vrc = 1.0, damp = 0.95;       // paper: velocity reduction coeff = 1, damping = 0.95
  double levy_scale = 1.0;             // paper uses raw L (scale 1); tunable via --levy-scale
  double lo = 0, hi = 0, sigma = 0;
};
