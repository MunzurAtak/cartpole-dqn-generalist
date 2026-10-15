"""Overall test score of every trained model (one dot per seed), grouped by strategy."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("results/test_scores.csv")
per_seed = df.groupby(["strategy", "seed"])["score"].mean().reset_index()
strategies = sorted(per_seed["strategy"].unique())
rng = np.random.default_rng(0)

plt.figure(figsize=(6, 4))
for i, s in enumerate(strategies):
    scores = per_seed.loc[per_seed["strategy"] == s, "score"]
    jitter = rng.uniform(
        -0.08, 0.08, len(scores)
    )  # spread dots horizontally so they don't overlap
    plt.scatter(np.full(len(scores), i) + jitter, scores, s=35, alpha=0.8)
    plt.hlines(
        scores.mean(), i - 0.25, i + 0.25, color="black", linewidth=2
    )  # strategy mean

plt.xticks(range(len(strategies)), strategies)
plt.ylabel("Average score over 30 pole lengths")
plt.title("Test score per training seed (black line = mean)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/per_seed_scores.png", dpi=200)
