#!/usr/bin/env python3
"""Compare seq / omp / cuda result CSVs: solution quality, runtime, speed-up, parity check."""
import sys, os, pandas as pd
files = [f for f in sys.argv[1:] if os.path.exists(f)]
if not files: sys.exit("usage: analyze.py seq.csv omp.csv cuda.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
keys = ["func", "dim", "pop", "iters", "shift"]
g = df.groupby(keys + ["version"]).agg(best_mean=("best", "mean"), best_std=("best", "std"),
                                       ms=("time_ms", "mean")).reset_index()
rows = []
for k, sub in g.groupby(keys, sort=False):
    s = sub.set_index("version")
    r = dict(zip(keys, k))
    for v in ("seq", "omp", "cuda"):
        if v in s.index:
            r[f"{v}_best(mean±std)"] = f"{s.loc[v,'best_mean']:.3e}±{s.loc[v,'best_std']:.1e}"
            r[f"{v}_ms"] = round(s.loc[v, "ms"], 2)
    if "seq" in s.index:
        for v in ("omp", "cuda"):
            if v in s.index: r[f"{v}_speedup"] = round(s.loc["seq", "ms"] / s.loc[v, "ms"], 2)
    rows.append(r)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(pd.DataFrame(rows).to_string(index=False))

# Parity: same seed => the three back-ends should reach (nearly) the same best value.
if "seq" in set(df.version):
    base = df[df.version == "seq"].set_index(keys + ["run"])["best"]
    for v in ("omp", "cuda"):
        if v in set(df.version):
            o = df[df.version == v].set_index(keys + ["run"])["best"]
            d = (o - base).abs().dropna()
            print(f"\nparity seq vs {v}: max |Δbest| = {d.max():.3e}  ({(d > 1e-6*(1+base.abs().reindex(d.index))).sum()} of {len(d)} runs differ by >1e-6 rel.)")
