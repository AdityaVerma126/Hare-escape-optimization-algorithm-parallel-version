// heo_main.h - common CLI + timing driver. Each back-end defines
//   static void   heo_backend_init();
//   static double heo_run(const Params&, uint64_t seed);   // returns best fitness
// and VERSION_NAME, then includes this file at the bottom.
#pragma once
#include "heo_common.h"
#include <chrono>

static void usage() {
  fprintf(stderr,
    "options: --func <id|all> --dim D --pop N --iters T --runs R --seed S\n"
    "         --shift  --levy-scale s  --no-header  --list\n");
}

int main(int argc, char** argv) {
  Params p; std::string func = "all"; bool header = true;
  for (int a = 1; a < argc; a++) {
    std::string s = argv[a];
    auto need = [&]() -> const char* {
      if (a + 1 >= argc) { fprintf(stderr, "missing value for %s\n", s.c_str()); exit(1); }
      return argv[++a]; };
    if (s == "--func") func = need();
    else if (s == "--dim") p.dim = atoi(need());
    else if (s == "--pop") p.pop = atoi(need());
    else if (s == "--iters") p.iters = atoi(need());
    else if (s == "--runs") p.runs = atoi(need());
    else if (s == "--seed") p.seed = strtoull(need(), 0, 10);
    else if (s == "--levy-scale") p.levy_scale = atof(need());
    else if (s == "--shift") p.shift = 1;
    else if (s == "--no-header") header = false;
    else if (s == "--list") { for (int k = 0; k < NFUNCS; k++) printf("%2d  %-18s [%g, %g]\n", k, FUNCS[k].name, FUNCS[k].lo, FUNCS[k].hi); return 0; }
    else { usage(); return 1; }
  }
  heo_backend_init();
  if (header) printf("version,func,dim,pop,iters,shift,run,best,time_ms\n");
  int f0 = 0, f1 = NFUNCS;
  if (func != "all") { f0 = atoi(func.c_str()); f1 = f0 + 1; if (f0 < 0 || f0 >= NFUNCS) { usage(); return 1; } }
  for (int fid = f0; fid < f1; fid++) {
    Params q = p; q.fid = fid; q.lo = FUNCS[fid].lo; q.hi = FUNCS[fid].hi;
    if (FUNCS[fid].fixed_dim) q.dim = FUNCS[fid].fixed_dim;
    q.shift = p.shift && FUNCS[fid].shiftable;
    q.sigma = levy_sigma_host();
    Params w = q; w.iters = 2; heo_run(w, 1);            // untimed warm-up (JIT, threads, CUDA ctx)
    for (int r = 0; r < p.runs; r++) {
      uint64_t seed = p.seed + 1000003ULL * (uint64_t)r;
      auto t0 = std::chrono::steady_clock::now();
      double f = heo_run(q, seed);
      auto t1 = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      printf("%s,%s,%d,%d,%d,%d,%d,%.12e,%.4f\n", VERSION_NAME, FUNCS[fid].name,
             q.dim, q.pop, q.iters, q.shift, r, f, ms);
      fflush(stdout);
    }
  }
  return 0;
}
