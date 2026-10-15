"""
Pole-length sampling strategies.

Each strategy decides which pole length is used for the next training episode.
This is the ONLY component that differs between experimental conditions.
"""

import numpy as np

MIN_LEN, MAX_LEN = 0.4, 1.8
DEFAULT_LEN = 0.5  # CartPole-v1 default pole length

STRATEGIES = ["baseline", "s1_extremes", "s2_extremes_mid", "s3_uniform"]


class FixedSetStrategy:
    """Sample a length uniformly from a fixed, discrete set at the start of each episode."""

    def __init__(self, lengths, rng):
        self.lengths = list(lengths)
        self.rng = rng

    def sample(self):
        return float(self.rng.choice(self.lengths))

    def validation_lengths(self):
        """Lengths used for checkpoint selection: exactly the training set."""
        return self.lengths


class UniformStrategy:
    """Sample a length from the continuous interval [low, high] (domain randomisation)."""

    def __init__(self, low, high, rng):
        self.low, self.high = low, high
        self.rng = rng

    def sample(self):
        return float(self.rng.uniform(self.low, self.high))

    def validation_lengths(self):
        """Five evenly spaced lengths covering the training interval."""
        return list(np.linspace(self.low, self.high, 5))


def make_strategy(name, rng):
    if name == "baseline":
        return FixedSetStrategy([DEFAULT_LEN], rng)
    if name == "s1_extremes":
        return FixedSetStrategy([MIN_LEN, MAX_LEN], rng)
    if name == "s2_extremes_mid":
        return FixedSetStrategy([MIN_LEN, 1.1, MAX_LEN], rng)
    if name == "s3_uniform":
        return UniformStrategy(MIN_LEN, MAX_LEN, rng)
    raise ValueError(f"Unknown strategy: {name}")
