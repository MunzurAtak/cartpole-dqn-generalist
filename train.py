"""
Train a DQN agent on CartPole-v1 using a given pole-length strategy.

Usage:
    python train.py --strategy s1_extremes --seeds 0 1 2
    python train.py --strategy baseline --seeds 0 --episodes 200   # quick smoke test
"""

import argparse
import json
import os
import random
import time

import gym
import numpy as np
import pandas as pd
import torch

from dqn import DQNAgent
from strategies import STRATEGIES, make_strategy

# Identical for all strategies, so only the pole-length distribution differs
CONFIG = {
    "episodes": 600,  # training budget per run
    "gamma": 0.99,  # discount factor
    "lr": 5e-4,  # Adam learning rate
    "batch_size": 64,
    "buffer_size": 50_000,  # replay buffer capacity
    "min_buffer": 1_000,  # transitions collected before learning starts
    "target_update_freq": 500,  # gradient steps between target-network syncs
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_frac": 0.5,  # epsilon reaches eps_end after this fraction of episodes
    "grad_clip": 10.0,  # max gradient norm
    "eval_every": 25,  # episodes between greedy validation runs
    "eval_episodes": 3,  # validation episodes per validation length
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def epsilon_at(episode, cfg):
    """Linear epsilon decay from eps_start to eps_end, then constant."""
    frac = min(1.0, episode / (cfg["eps_decay_frac"] * cfg["episodes"]))
    return cfg["eps_start"] + frac * (cfg["eps_end"] - cfg["eps_start"])


def run_episode(env, agent, length, epsilon, learn):
    """Play one episode with the given pole length. If learn=True, store transitions and train."""
    env.unwrapped.length = length
    state, _ = env.reset()
    total_reward, done = 0.0, False

    while not done:
        action = agent.act(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        if learn:
            agent.buffer.push(state, action, reward, next_state, terminated)
            agent.update()
        state = next_state
        total_reward += reward
        done = terminated or truncated

    return total_reward


def validate(env, agent, lengths, n_episodes):
    """Greedy performance on the strategy's own validation lengths (NOT the test script)."""
    scores = [
        run_episode(env, agent, L, 0.0, learn=False)
        for L in lengths
        for _ in range(n_episodes)
    ]
    return float(np.mean(scores))


def train(strategy_name, seed, cfg):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    strategy = make_strategy(strategy_name, rng)

    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    val_env = gym.make("CartPole-v1")
    val_env.reset(seed=seed + 10_000)

    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg)

    run_name = f"{strategy_name}_seed{seed}"
    os.makedirs("weights", exist_ok=True)
    os.makedirs("results/train_logs", exist_ok=True)

    log, best_val, start = [], -np.inf, time.time()

    for ep in range(cfg["episodes"]):
        length = strategy.sample()
        eps = epsilon_at(ep, cfg)
        ret = run_episode(env, agent, length, eps, learn=True)
        row = {
            "episode": ep,
            "length": length,
            "epsilon": eps,
            "return": ret,
            "val_return": np.nan,
        }

        # Periodic validation; keep the best checkpoint
        if (ep + 1) % cfg["eval_every"] == 0:
            val = validate(
                val_env, agent, strategy.validation_lengths(), cfg["eval_episodes"]
            )
            row["val_return"] = val
            if val >= best_val:
                best_val = val
                torch.save(agent.q_net.state_dict(), f"weights/{run_name}.pth")
            print(
                f"[{run_name}] ep {ep + 1}/{cfg['episodes']} | eps {eps:.2f} | "
                f"return {ret:.0f} | val {val:.1f} | best {best_val:.1f} | {time.time() - start:.0f}s",
                flush=True,
            )

        log.append(row)

    pd.DataFrame(log).to_csv(f"results/train_logs/{run_name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=CONFIG["episodes"])
    args = parser.parse_args()

    torch.set_num_threads(
        1
    )  # avoids CPU oversubscription when running several trainings in parallel
    cfg = {**CONFIG, "episodes": args.episodes}

    os.makedirs("results", exist_ok=True)
    with open("results/config.json", "w") as f:
        json.dump(cfg, f, indent=2)  # for reproducibility

    for seed in args.seeds:
        train(args.strategy, seed, cfg)
