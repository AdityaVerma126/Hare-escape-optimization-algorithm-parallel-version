# HEO Sequential / OpenMP / CUDA: final report (profile `final`, finished 2026-11-15T03:13:50)

## 1. Hardware configuration
* CPU: Intel(R) Xeon(R) CPU @ 2.00GHz | visible CPU threads: 2 | RAM: 12.7 GB
* GPU: Tesla T4 | 15360 MiB | compute capability 7.5 | driver 580.82.07

## 2. Software configuration
* Python 3.13.16 | Linux-6.6.159+-x86_64-with-glibc2.39
* Compiler: g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 (OpenMP 201511) | CUDA toolkit: 13.0
* Build flags: CPU `-O3 -std=c++17 -ffp-contract=off` (+ `-fopenmp`); CUDA `-O3 -std=c++17 -arch=sm_75 --fmad=false`

## 3. Algorithm parameters
* Update (paper Eq. 4): `X' = clamp(X + c_t V + L (X+V) r)`, `V' = 0.95 V`; velocity-reduction coefficient 1, damping 0.95
* Default coefficient `c_t = (T-t+1)/T` (documented interpretation); the paper's literal `Maxit/t` is available with `--scale literal` and was **not** used for the results below
* L: Mantegna Levy step, beta = 1.5, unscaled (sigma_u = 0.6966); r ~ U(0,1); bounds handled by clamping; elitist merge-and-keep-best-N selection (parents win ties)
* Initial positions and velocities uniform in the function bounds; counter-based RNG (splitmix64)

## 4. Benchmark methodology
* Configurations (D, N, T): (30, 256, 200); (30, 1024, 500); (100, 256, 500); (100, 1024, 200)
* Independent runs per configuration: 20 (meets the 20-run requirement) | shifted-optimum study: 20 runs on one configuration
* Matched seeds: run k uses 42 + 1000003 k in every back-end | OpenMP threads: 8 | back-ends: seq, omp, cuda
* Timing: end-to-end wall clock per run, and compute-only loop time (CUDA events on the GPU); untimed warm-up of identical size; total benchmark time 2706.4 s

## 5. Correctness validation
* `--selftest` on every executable: RNG statistics, Levy sigma, benchmark optima, update rule, merge selection (see Section 3 output)
* OpenMP vs sequential over 1220 matched-seed runs: 1220 bit-identical, max abs diff 0, max relative diff 0, runs above 1e-6: 0
* CUDA GPU vs sequential over 1220 matched-seed runs: 572 bit-identical, max abs diff 0.000579, max relative diff 0.00029, runs above 1e-6: 4

## 6. Runtime comparison, speedup and efficiency (scalable functions, unshifted; geometric mean over functions)
|                             |   end_to_end |   compute_only |   efficiency |   overhead |
|:----------------------------|-------------:|---------------:|-------------:|-----------:|
| ('D100_N1024_T200', 'cuda') |       34.854 |         34.872 |      nan     |      0.001 |
| ('D100_N1024_T200', 'omp')  |        1.204 |          1.204 |        0.15  |      0     |
| ('D100_N256_T500', 'cuda')  |       14.813 |         14.819 |      nan     |      0     |
| ('D100_N256_T500', 'omp')   |        1.186 |          1.186 |        0.148 |      0     |
| ('D30_N1024_T500', 'cuda')  |       29.639 |         29.654 |      nan     |      0.001 |
| ('D30_N1024_T500', 'omp')   |        1.131 |          1.131 |        0.141 |      0     |
| ('D30_N256_T200', 'cuda')   |       10.435 |         10.45  |      nan     |      0.001 |
| ('D30_N256_T200', 'omp')    |        1.117 |          1.117 |        0.14  |      0     |

(efficiency = speedup / OpenMP threads; not defined for CUDA)

## 7. Statistical analysis
* Wilcoxon signed-rank, final fitness: 122 comparisons, 68 identical or untestable, 2/54 testable with p < 0.05 (uncorrected)
* Wilcoxon signed-rank, runtime: 122 comparisons, 0 identical or untestable, 117/122 testable with p < 0.05 (uncorrected)
* Friedman, fitness: chi2 = 1.00, p = 0.607 over 1220 blocks | runtime: chi2 = 1838.50, p = 0 over 1220 blocks

## 8. Not executed / caveats about this run
* The paper's quality claims (29 competitors, CEC 2015 shifted/rotated functions, engineering problems) are **not** reproduced here

## 9. Limitations
* Several equations of the paper are ambiguous; the choices (A1-A9 in `heo_common.h`) affect fitness but not the parallel-performance conclusions
* The update term scales with the position itself, which favours optima at the origin (see the shifted-optimum figure)
* Timings reflect this hardware and these sizes; one-time allocation is excluded by the warm-up; the sort and gather keep part of the work memory-bound
* CUDA agrees with the CPU only up to device math-library rounding; a changed selection order can, rarely, change a trajectory
* Wilcoxon p-values are not corrected for multiple comparisons

## 10. Future improvements
* CUDA Graphs or a persistent kernel to remove per-iteration launch latency; a device-side bitonic sort for small N
* Float (`-DUSE_FLOAT`) and tensor-friendly layouts; multi-GPU and CPU+GPU hybrid populations
* CEC 2015 functions with the official shift/rotation data, constrained engineering problems, adaptive parameter tuning