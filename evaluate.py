"""
Evaluate all trained networks in weights/ with the official test protocol.
Uses test_pole_length() from the provided test script UNCHANGED.

Usage:
    python evaluate.py                                    # all models, 10 runs per length
    python evaluate.py --pattern baseline_seed0 --runs 1  # quick check of one model
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import gym
import matplotlib

matplotlib.use("Agg")  # save plots to file without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from test_script import QNetwork, test_pole_length

POLE_LENGTHS = np.linspace(0.4, 1.8, 30)  # same grid as the official test script


def evaluate_model(path, runs):
    """Run the official protocol on one model; returns one row per (length, run)."""
    torch.set_num_threads(1)
    name = os.path.splitext(os.path.basename(path))[0]
    strategy, seed = name.rsplit("_seed", 1)

    model = QNetwork(4, 2)  # CartPole: 4 state variables, 2 actions
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()

    rows = []
    for length in POLE_LENGTHS:
        for run in range(runs):
            env = gym.make("CartPole-v1")  # fresh env per run
            env.unwrapped.length = length
            with torch.no_grad():
                score = test_pole_length(env, model)
            rows.append(
                {
                    "strategy": strategy,
                    "seed": int(seed),
                    "length": round(float(length), 2),
                    "run": run,
                    "score": score,
                }
            )
    print(f"finished {name}", flush=True)
    return rows


def bar_plot(df, strategy, out_path):
    """Required plot: mean +/- std per pole length, overall average in the title."""
    stats = df.groupby("length")["score"].agg(mean="mean", std=lambda s: s.std(ddof=0))
    overall = stats["mean"].mean()

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(stats)), stats["mean"], yerr=stats["std"], capsize=5, alpha=0.7)
    plt.xticks(range(len(stats)), stats.index, rotation=45)
    plt.xlabel("Pole length")
    plt.ylabel("Episode length")
    plt.title(f"{strategy}: average score over all pole lengths = {overall:.0f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def comparison_plot(df, out_path):
    """All strategies in one figure: mean score per pole length."""
    plt.figure(figsize=(8, 5))
    for strategy, d in df.groupby("strategy"):
        m = d.groupby("length")["score"].mean()
        plt.plot(m.index, m.values, marker="o", markersize=3, label=strategy)
    plt.xlabel("Pole length")
    plt.ylabel("Mean episode length")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def holm_correction(pvalues):
    """Holm-Bonferroni correction for multiple comparisons (input sorted ascending)."""
    m, adjusted, running_max = len(pvalues), [], 0.0
    for k, p in enumerate(pvalues):
        running_max = max(running_max, min(1.0, (m - k) * p))
        adjusted.append(running_max)
    return adjusted


def statistics(df):
    # Overall score per (strategy, seed) -> variability between training runs
    per_seed = df.groupby(["strategy", "seed"])["score"].mean()
    summary = per_seed.groupby(level="strategy").agg(["mean", "std", "count"])

    # Mean score per pole length per strategy -> pairs for the Wilcoxon test (n = 30)
    per_length = df.groupby(["length", "strategy"])["score"].mean().unstack("strategy")
    names = list(per_length.columns)

    tests = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            stat, p = wilcoxon(per_length[a], per_length[b])
            tests.append(
                {
                    "A": a,
                    "B": b,
                    "median_diff_A_minus_B": float(
                        (per_length[a] - per_length[b]).median()
                    ),
                    "W": float(stat),
                    "p": float(p),
                }
            )

    tests = pd.DataFrame(tests).sort_values("p").reset_index(drop=True)
    tests["p_holm"] = holm_correction(tests["p"].tolist())
    tests["significant_0.05"] = tests["p_holm"] < 0.05
    return summary, tests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="*", help="glob for weights/<pattern>.pth")
    parser.add_argument(
        "--runs", type=int, default=10, help="test runs per pole length"
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    paths = sorted(glob.glob(f"weights/{args.pattern}.pth"))
    print(f"Evaluating {len(paths)} models: {[os.path.basename(p) for p in paths]}")
    os.makedirs("results/plots", exist_ok=True)

    # Evaluate models in parallel (one process per model)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = ex.map(evaluate_model, paths, [args.runs] * len(paths))
        df = pd.DataFrame([row for rows in results for row in rows])

    df.to_csv("results/test_scores.csv", index=False)

    for strategy, d in df.groupby("strategy"):
        bar_plot(d, strategy, f"results/plots/barplot_{strategy}.png")
    comparison_plot(df, "results/plots/comparison.png")

    if df["strategy"].nunique() > 1:
        summary, tests = statistics(df)
        summary.to_csv("results/summary.csv")
        tests.to_csv("results/wilcoxon_tests.csv", index=False)
        print("\nOverall score per strategy (mean/std over seeds):\n", summary)
        print("\nPairwise Wilcoxon signed-rank tests (Holm-corrected):\n", tests)
