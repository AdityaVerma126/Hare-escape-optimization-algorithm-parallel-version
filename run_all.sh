#!/usr/bin/env bash
# Usage: ./run_all.sh      (env: RUNS ITERS POP DIM, OMP_NUM_THREADS)
set -e
RUNS=${RUNS:-20}; ITERS=${ITERS:-500}; POP=${POP:-30}; DIM=${DIM:-30}
mkdir -p results
for v in seq omp cuda; do
  [ -x ./heo_$v ] || { echo "skip $v (not built)"; continue; }
  echo "== $v: benchmark suite =="
  ./heo_$v --func all --dim $DIM --pop $POP --iters $ITERS --runs $RUNS         > results/bench_${v}.csv
  ./heo_$v --func all --dim $DIM --pop $POP --iters $ITERS --runs $RUNS --shift > results/bench_${v}_shift.csv
  echo "== $v: scaling sweep (Rastrigin, D=100) =="
  for n in 100 1000 10000 100000; do
    ./heo_$v --func 3 --dim 100 --pop $n --iters 200 --runs 5 --no-header >> results/scale_${v}.csv.tmp
  done
  { echo "version,func,dim,pop,iters,shift,run,best,time_ms"; cat results/scale_${v}.csv.tmp; } > results/scale_${v}.csv
  rm results/scale_${v}.csv.tmp
done
python3 analyze.py results/bench_seq.csv results/bench_omp.csv results/bench_cuda.csv
python3 analyze.py results/bench_seq_shift.csv results/bench_omp_shift.csv results/bench_cuda_shift.csv
python3 analyze.py results/scale_seq.csv results/scale_omp.csv results/scale_cuda.csv
