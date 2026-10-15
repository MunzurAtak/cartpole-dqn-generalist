# CartPole DQN Generalist

Group Assignment 3 for _Introduction to Reinforcement Learning_ (VU Amsterdam, 2026).

We train a Deep Q-Network (DQN) with the fixed "standard" network from the test script to balance
CartPole-v1 for **any pole length between 0.4 and 1.8**. The agent never observes the pole length, so it
has to learn one policy that works for every length. We compare four strategies for choosing which pole
lengths the agent sees during training, to answer this question: **how does the distribution of training
pole lengths affect generalisation across the full range?**

## Training strategies

The strategies differ **only** in which pole length is used at the start of each training episode.
All other settings (network, hyperparameters, 600 episodes, seeds 0–4) are identical.

| Strategy                           | Name in code      | Pole lengths during training       |
| ---------------------------------- | ----------------- | ---------------------------------- |
| Baseline                           | `baseline`        | Default length only: 0.5           |
| S1: Extremes                       | `s1_extremes`     | Random choice from {0.4, 1.8}      |
| S2: Extremes + midpoint            | `s2_extremes_mid` | Random choice from {0.4, 1.1, 1.8} |
| S3: Uniform (domain randomisation) | `s3_uniform`      | Uniformly random in [0.4, 1.8]     |

## Key results

Evaluated with the official protocol: 30 pole lengths in [0.4, 1.8], 10 test runs per length,
wind attacks from step 500 and increasing force from step 1000.

| Strategy                | Average score over all pole lengths (mean ± std over 5 seeds) |
| ----------------------- | ------------------------------------------------------------- |
| Baseline                | 2,333 ± 1,516                                                 |
| S1: Extremes            | 5,551 ± 5,570                                                 |
| S2: Extremes + midpoint | 6,597 ± 7,826                                                 |
| **S3: Uniform**         | **10,873 ± 6,096**                                            |

- **Final submitted network:** `weights/s3_uniform_seed1.pth`. It scores **19,412** with the official
  `test_script.py`, and `test_script.py` is already set to this file.
- **Per pole length (Wilcoxon signed-rank, n = 30, Holm-corrected):** all pairwise differences are
  significant (`results/wilcoxon_tests.csv`).
- **Per seed (Mann-Whitney U, n = 5 vs 5):** no difference is significant at α = 0.05. The closest is
  S3 vs baseline, p = 0.056 (`results/seed_level_tests.csv`). Variation between training seeds is very
  large, so the ranking is consistent across pole lengths but uncertain across training runs.

## Setup

Requires **Python 3.10–3.12**. Python 3.13+ does not work, because gym 0.26 needs NumPy < 2,
which has no pre-built packages for newer Python versions.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
$env:PYTHONWARNINGS="ignore"    # optional: hides gym's deprecation warnings
```

Always run the scripts **from this folder**, because all paths are relative to it.

## Usage

```powershell
# 1. Train one strategy for several seeds (about 1-2 minutes per seed on a laptop CPU)
python train.py --strategy s3_uniform --seeds 0 1 2 3 4
#    strategies: baseline | s1_extremes | s2_extremes_mid | s3_uniform

# 2. Evaluate all networks in weights/ with the official test protocol + plots + statistics
python evaluate.py
python evaluate.py --pattern s3_uniform_seed1 --runs 1   # quick check of one model

# 3. Extra analysis (reads results/test_scores.csv, no re-evaluation needed)
python seed_level_tests.py
python plot_per_seed.py

# 4. Official test script on the final network (set trained_nn inside the file)
python test_script.py
```

## What every file and folder is

### Code

| File                  | What it does                                                                                                                                                                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `test_script.py`      | **Provided by the course; unchanged** except `trained_nn`, which points to our final network. Contains the fixed `QNetwork` and the official `test_pole_length()` function.                                                                      |
| `dqn.py`              | DQN components: `ReplayBuffer` (experience replay) and `DQNAgent` (ε-greedy actions, TD update with a target network, Huber loss, gradient clipping). Imports `QNetwork` from `test_script.py` so the weights always match the official network. |
| `strategies.py`       | The pole-length sampling strategies (the independent variable). Each strategy has `sample()` (length for the next episode) and `validation_lengths()` (lengths used to select the best checkpoint).                                              |
| `train.py`            | Training loop. Contains `CONFIG` with all hyperparameters. Every 25 episodes it validates greedily on the strategy's **own** training lengths and saves the best checkpoint to `weights/`. The test script is never used during training.        |
| `evaluate.py`         | Runs the official test protocol on every network in `weights/`, by calling `test_pole_length()` from `test_script.py` unchanged. Saves all scores, the bar plots, the comparison plot and the Wilcoxon tests. Models are evaluated in parallel.  |
| `seed_level_tests.py` | Mann-Whitney U tests on the overall score per trained model (5 seeds per strategy).                                                                                                                                                              |
| `plot_per_seed.py`    | Plot with one dot per trained model, which shows the variation between seeds.                                                                                                                                                                    |
| `requirements.txt`    | Package versions (gym 0.26.2, NumPy < 2, PyTorch, pandas, matplotlib, scipy, openpyxl).                                                                                                                                                          |

### `weights/`: trained networks

`<strategy>_seed<k>.pth` (20 files: 4 strategies × 5 seeds). Each is the `state_dict` of the standard
`QNetwork` at the checkpoint with the best validation score during training. These are the files the
test script loads.

### `results/`: everything the report needs

| File / folder                       | Contents                                                                                                                                                                                                   |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.json`                       | Exact hyperparameters used for all runs (for the Methods section).                                                                                                                                         |
| `train_logs/<strategy>_seed<k>.csv` | One row per training episode: `episode`, `length` (pole length used), `epsilon`, `return`, and `val_return` (every 25th episode). Use these for learning curves.                                           |
| `test_scores.csv`                   | **Raw test data**: one row per test episode (`strategy`, `seed`, `length`, `run`, `score`). 20 models × 30 lengths × 10 runs = 6,000 rows. Every other result is calculated from this file.                |
| `summary.csv`                       | Overall score per strategy: mean, std and count over the 5 seeds.                                                                                                                                          |
| `wilcoxon_tests.csv`                | Pairwise Wilcoxon signed-rank tests, paired per pole length (n = 30), with raw and Holm-corrected p-values.                                                                                                |
| `seed_level_tests.csv`              | Pairwise Mann-Whitney U tests on the overall score per seed (n = 5 vs 5).                                                                                                                                  |
| `plots/barplot_<strategy>.png`      | Required plot per strategy: mean ± std per pole length over 5 seeds × 10 runs, overall average in the title.                                                                                               |
| `plots/comparison.png`              | All strategies in one figure: mean score per pole length.                                                                                                                                                  |
| `plots/per_seed_scores.png`         | Overall score of each of the 20 trained models, grouped by strategy.                                                                                                                                       |
| `test_script_plots/`                | Output of the **official** `test_script.py`: `s3_uniform_seed1` (final network, 19,412) and `s2_extremes_mid_seed0` (18,085). These are bar plots and Excel files in exactly the format of the assignment. |

## Notes for the report

- **Error bars** in `barplot_*.png` are the std over 5 seeds × 10 runs, so they include the variation
  _between_ trained models, which is the main source of spread. They go below 0 because scores are
  heavily skewed.
- **Test protocol quirk:** in `test_pole_length()`, `force_mag` is set to 75 at step 500 and is never
  reset, so steps 500–1000 use 7.5× the normal force. After step 1000 the force is 25 + 0.01·t, which
  is about 35 at first and does not reach 75 again until step 5000.
- **Pole length:** `env.unwrapped.length` changes only `length`. The cached `polemass_length` keeps its
  default value. This applies to training and testing alike.
- **Training budget** is in episodes, not steps. Strategies that train on short poles have shorter
  episodes and therefore fewer environment steps in total (see `train_logs/`).
- **DQN was unstable** in several runs (collapses after reaching 500, long plateaus around 100–150).
  Two models failed in testing: `s1_extremes_seed1` and `s2_extremes_mid_seed3`. These are exactly the
  runs whose best validation score never got above about 350.

## Group

Group 1: Munzur Atak, Max Leenman, Arin Gupta
