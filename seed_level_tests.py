"""Seed-level comparison: Mann-Whitney U on each run's overall score (n = 5 per strategy)."""

from itertools import combinations

import pandas as pd
from scipy.stats import mannwhitneyu

df = pd.read_csv("results/test_scores.csv")
per_seed = df.groupby(["strategy", "seed"])["score"].mean()

rows = []
for a, b in combinations(sorted(df["strategy"].unique()), 2):
    u, p = mannwhitneyu(per_seed.loc[a], per_seed.loc[b], alternative="two-sided")
    rows.append({"A": a, "B": b, "U": u, "p": p})

out = pd.DataFrame(rows).sort_values("p")
print(out)
out.to_csv("results/seed_level_tests.csv", index=False)
