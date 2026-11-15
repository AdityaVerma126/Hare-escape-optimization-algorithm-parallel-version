// heo_main.h - common CLI + timing driver, included at the bottom of every back-end. A back-end defines
//   #define VERSION_NAME "seq"|"omp"|"cuda"
//   static int    heo_backend_init(int threads);   // returns the thread count to report (0 = not applicable)
//   static double heo_run(const Params&, uint64_t seed, RunInfo* info);   // returns best fitness
// and then includes this file. stdout carries ONLY the CSV (or --list / --selftest text); every
// diagnostic goes to stderr, so a notebook can parse stdout directly.
#pragma once
#include "heo_common.h"
#include <chrono>
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <random>

struct SelfAcc { const double* p; HD double operator()(int i) const { return p[i]; } };

static void usage() {
  fprintf(stderr,
    "usage: heo_%s [options]\n"
    "  --func <id|name|all>   benchmark function (default all; see --list)\n"
    "  --dim D --pop N --iters T --runs R --seed S     (defaults 30 / 30 / 500 / 10 / 42)\n"
    "  --threads K            OpenMP threads (ignored by seq and cuda; default: OMP_NUM_THREADS / all cores)\n"
    "  --shift                move the optimum away from the origin (functions that support it)\n"
    "  --scale decay|literal  position coefficient: (T-t+1)/T (default) or Maxit/t as printed in the paper\n"
    "  --levy-scale s         multiply the Levy step (default 1)\n"
    "  --block B              CUDA threads per block, multiple of 32 (default 256; ignored on CPU)\n"
    "  --history <file>       also write best-so-far per iteration for run 0 (untimed extra run)\n"
    "  --no-header  --list  --selftest  --help\n"
    "run k uses seed S + 1000003*k in every back-end; CSV: version,func,dim,pop,iters,shift,run,seed,threads,best,time_ms,compute_ms\n",
    VERSION_NAME);
}
[[noreturn]] static void die(const std::string& msg) {
  fprintf(stderr, "error: %s\n", msg.c_str()); usage(); exit(2);
}
static long long parse_ll(const std::string& opt, const char* s, long long lo, long long hi) {
  char* end = nullptr; errno = 0; long long v = strtoll(s, &end, 10);
  if (end == s || *end != '\0' || errno) die(opt + ": '" + s + "' is not an integer");
  if (v < lo || v > hi) die(opt + ": " + s + " is out of range [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
  return v;
}
static double parse_dbl(const std::string& opt, const char* s) {
  char* end = nullptr; errno = 0; double v = strtod(s, &end);
  if (end == s || *end != '\0' || errno || !std::isfinite(v)) die(opt + ": '" + s + "' is not a finite number");
  return v;
}
static std::string lower(std::string s) { for (auto& c : s) c = (char)std::tolower((unsigned char)c); return s; }
// accepts a numeric id, the full name (any case) or the short tag before '_' (e.g. F19)
static int find_func(const std::string& s) {
  char* end = nullptr; long v = strtol(s.c_str(), &end, 10);
  if (end != s.c_str() && *end == '\0') return (v >= 0 && v < NFUNCS) ? (int)v : -1;
  std::string l = lower(s);
  for (int k = 0; k < NFUNCS; k++) {
    std::string n = lower(FUNCS[k].name), tag = n.substr(0, n.find('_'));
    if (l == n || l == tag) return k;
  }
  return -1;
}

// ------------------------------------------------------------------ --selftest (host-side checks)
static int heo_selftest() {
  int fails = 0;
  auto chk = [&](bool ok, const char* what, double val) {
    printf("%s  %-52s %.6g\n", ok ? "PASS" : "FAIL", what, val); if (!ok) fails++; };
  { const int n = 1000000; double s1 = 0, s2 = 0, mn = 1, mx = 0;
    for (int i = 0; i < n; i++) { double u = u01(7, (uint64_t)(i / 1000), (uint64_t)(i % 1000), 0);
      s1 += u; s2 += u * u; mn = std::min(mn, u); mx = std::max(mx, u); }
    double m = s1 / n, v = s2 / n - m * m;
    chk(std::fabs(m - 0.5) < 2e-3, "u01 mean ~ 0.5", m);
    chk(std::fabs(v - 1.0 / 12) < 2e-3, "u01 variance ~ 1/12", v);
    chk(mn > 0.0 && mx < 1.0, "u01 stays inside the open interval (0,1)", mn); }
  { double sg = levy_sigma_host(); chk(std::fabs(sg - 0.6966) < 1e-3, "Levy sigma_u(beta=1.5) ~ 0.6966", sg);
    int bad = 0; for (int i = 0; i < 200000; i++) { double l = levy_step(11, 1 + i % 50, (uint64_t)i, sg); if (!std::isfinite(l)) bad++; }
    chk(bad == 0, "Levy steps are finite (count of non-finite)", bad); }
  { double z[8]; auto ev = [&](int id, int D) { return bench_eval(id, D, SelfAcc{z}, 0.0); };
    for (int i = 0; i < 8; i++) { z[i] = 0; }
    chk(ev(0, 8) == 0 && ev(3, 8) == 0 && std::fabs(ev(4, 8)) < 1e-12 && ev(5, 8) == 0, "f(0)=0 for StepSphere/Rastrigin/Ackley/Griewank", ev(4, 8));
    chk(ev(7, 2) == 0 && ev(9, 2) == 0 && ev(12, 2) == 0 && ev(1, 8) == 0, "f(0)=0 for Bohachevsky/Matyas/Schaffer2/Quartic(no noise)", 0);
    for (int i = 0; i < 8; i++) { z[i] = 420.9687462275036; }
    chk(std::fabs(ev(2, 8)) < 1e-9, "Schwefel at 420.9687 ~ 0", ev(2, 8));
    for (int i = 0; i < 8; i++) { z[i] = 1; }
    chk(ev(6, 8) == 0, "Rosenbrock at (1,..,1) = 0", ev(6, 8));
    z[0] = 1; z[1] = 3;                                   chk(ev(8, 2) == 0, "Booth at (1,3) = 0", ev(8, 2));
    z[0] = HEO_PI; z[1] = HEO_PI;                         chk(std::fabs(ev(10, 2) + 1) < 1e-12, "Easom at (pi,pi) = -1", ev(10, 2));
    z[0] = 3; z[1] = 0.5;                                 chk(ev(11, 2) == 0, "Beale at (3,0.5) = 0", ev(11, 2));
    z[0] = 0.08984201; z[1] = -0.71265640;                chk(std::fabs(ev(13, 2) + 1.0316285) < 1e-6, "Six-hump camel at its minimiser", ev(13, 2)); }
  { real xn, vn; heo_step(90, 50, 1, 10, 1.0, 0.95, 0, -100, 100, 0.7f, 0.3f, &xn, &vn);
    chk(xn == (real)100 && std::fabs((double)vn - 47.5) < 1e-9, "heo_step clamps position, damps velocity", (double)vn);
    heo_step(1.5, -2.0, 3, 10, 1.0, 0.95, 1, -100, 100, 0.7, 0.3, &xn, &vn);
    chk(std::fabs((double)xn - (1.5 + (10.0 / 3.0) * -2.0 + 0.7 * (1.5 - 2.0) * 0.3)) < 1e-12, "literal Maxit/t equals paper Eq.(4)", (double)xn); }
  { std::mt19937 g(1); long bad = 0, trials = 0;
    for (int tr = 0; tr < 2000; tr++) {
      int n = 1 + (int)(g() % 40); std::vector<double> A(n), B(n);
      for (auto& x : A) x = (double)(g() % 8);
      for (auto& x : B) x = (double)(g() % 8);
      std::sort(A.begin(), A.end()); std::sort(B.begin(), B.end());
      std::vector<std::pair<bool, int>> ref; int i = 0, j = 0;
      while ((int)ref.size() < n) { if (i < n && (j >= n || A[i] <= B[j])) ref.push_back({true, i++}); else ref.push_back({false, j++}); }
      for (int k = 0; k < n; k++) { bool fa; int pos;
        merge_select(k, n, [&](int q) { return A[q]; }, [&](int q) { return B[q]; }, &fa, &pos);
        trials++; if (fa != ref[k].first || pos != ref[k].second) bad++; } }
    chk(bad == 0, "merge_select == reference merge (mismatches)", (double)bad); }
  printf("%s (%d failure%s)\n", fails ? "SELFTEST FAILED" : "SELFTEST PASSED", fails, fails == 1 ? "" : "s");
  return fails ? 1 : 0;
}

int main(int argc, char** argv) {
  Params p; std::string func = "all", hist_file; bool header = true, dim_given = false; int threads_req = 0;
  for (int a = 1; a < argc; a++) {
    std::string s = argv[a];
    auto need = [&]() -> const char* { if (a + 1 >= argc) die("missing value for " + s); return argv[++a]; };
    if (s == "--func") func = need();
    else if (s == "--dim") { p.dim = (int)parse_ll(s, need(), 1, 100000000); dim_given = true; }
    else if (s == "--pop") p.pop = (int)parse_ll(s, need(), 1, 100000000);
    else if (s == "--iters") p.iters = (int)parse_ll(s, need(), 1, 1000000000);
    else if (s == "--runs") p.runs = (int)parse_ll(s, need(), 1, 1000000);
    else if (s == "--seed") p.seed = (uint64_t)parse_ll(s, need(), 0, 9223372036854775807LL);
    else if (s == "--threads") threads_req = (int)parse_ll(s, need(), 1, 4096);
    else if (s == "--levy-scale") p.levy_scale = parse_dbl(s, need());
    else if (s == "--block") p.block = (int)parse_ll(s, need(), 32, 1024);
    else if (s == "--scale") { std::string v = lower(need());
      if (v == "decay") p.scale_mode = 0; else if (v == "literal") p.scale_mode = 1; else die("--scale expects decay or literal"); }
    else if (s == "--history") hist_file = need();
    else if (s == "--shift") p.shift = 1;
    else if (s == "--no-header") header = false;
    else if (s == "--help" || s == "-h") { usage(); return 0; }
    else if (s == "--selftest") return heo_selftest();
    else if (s == "--list") {
      printf("id,name,lo,hi,fmin,fixed_dim,shiftable\n");
      for (int k = 0; k < NFUNCS; k++) printf("%d,%s,%g,%g,%.10g,%d,%d\n", k, FUNCS[k].name, FUNCS[k].lo, FUNCS[k].hi, FUNCS[k].fmin, FUNCS[k].fixed_dim, FUNCS[k].shiftable);
      return 0; }
    else die("unknown option '" + s + "'");
  }
  { std::string err = heo_validate(p); if (!err.empty()) die(err); }
  int f0 = 0, f1 = NFUNCS;
  if (func != "all") { f0 = find_func(func); if (f0 < 0) die("unknown function '" + func + "' (use --list)"); f1 = f0 + 1; }
  FILE* hf = nullptr;
  if (!hist_file.empty()) {
    hf = fopen(hist_file.c_str(), "w"); if (!hf) die("cannot open history file '" + hist_file + "'");
    fprintf(hf, "version,func,dim,pop,iters,shift,seed,iter,best\n");
  }
  int threads = heo_backend_init(threads_req);
  if (header) printf("version,func,dim,pop,iters,shift,run,seed,threads,best,time_ms,compute_ms\n");
  for (int fid = f0; fid < f1; fid++) {
    Params q = p; q.fid = fid; q.lo = FUNCS[fid].lo; q.hi = FUNCS[fid].hi;
    if (FUNCS[fid].fixed_dim) {
      if (dim_given && p.dim != FUNCS[fid].fixed_dim)
        fprintf(stderr, "note: %s is %d-dimensional by definition; --dim %d ignored for it\n", FUNCS[fid].name, FUNCS[fid].fixed_dim, p.dim);
      q.dim = FUNCS[fid].fixed_dim;
    }
    if (p.shift && !FUNCS[fid].shiftable) fprintf(stderr, "note: %s does not support --shift; running unshifted\n", FUNCS[fid].name);
    q.shift = p.shift && FUNCS[fid].shiftable;
    q.sigma = levy_sigma_host();
    Params w = q; w.iters = 2; heo_run(w, 1, nullptr);                // untimed warm-up at the SAME size (workspace, JIT, threads, CUDA context)
    if (hf) {                                                         // untimed convergence-history run: same seed as run 0
      std::vector<double> h; RunInfo hi; hi.history = &h; heo_run(q, p.seed, &hi);
      for (size_t t = 0; t < h.size(); t++)
        fprintf(hf, "%s,%s,%d,%d,%d,%d,%llu,%zu,%.17g\n", VERSION_NAME, FUNCS[fid].name, q.dim, q.pop, q.iters, q.shift, (unsigned long long)p.seed, t, h[t]);
    }
    for (int r = 0; r < p.runs; r++) {
      uint64_t seed = p.seed + 1000003ULL * (uint64_t)r;
      RunInfo info;
      auto t0 = std::chrono::steady_clock::now();
      double f = heo_run(q, seed, &info);
      auto t1 = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      if (!std::isfinite(f)) fprintf(stderr, "warning: non-finite best fitness (func %s, run %d)\n", FUNCS[fid].name, r);
      printf("%s,%s,%d,%d,%d,%d,%d,%llu,%d,%.17g,%.4f,", VERSION_NAME, FUNCS[fid].name, q.dim, q.pop, q.iters, q.shift, r,
             (unsigned long long)seed, threads, f, ms);
      if (info.compute_ms >= 0) printf("%.4f\n", info.compute_ms); else printf("nan\n");
      fflush(stdout);
    }
  }
  if (hf) fclose(hf);
  return 0;
}
