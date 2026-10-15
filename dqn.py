"""
Core DQN components: experience replay buffer and the DQN agent.

The Q-network architecture is imported from the provided test script so the
trained weights are guaranteed to be compatible with the evaluation code.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from test_script import QNetwork


class ReplayBuffer:
    """
    Fixed-size FIFO buffer storing transitions
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, terminated):
        """
        Add a transition to the buffer
        """
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminated = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(terminated, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN agent interacting with the environment
    """

    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim
        self.gamma = cfg["gamma"]
        self.batch_size = cfg["batch_size"]
        self.min_buffer = cfg["min_buffer"]
        self.target_update_freq = cfg["target_update_freq"]
        self.grad_clip = cfg["grad_clip"]

        # Online network (trained) and target network (frozen copy used for TD targets)
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg["lr"])
        self.buffer = ReplayBuffer(cfg["buffer_size"])
        self.n_updates = 0

    def act(self, state, epsilon):
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return self.q_net(s).argmax().item()

    def update(self):
        """One gradient step on a sampled mini-batch. Returns the loss (or None if buffer too small)."""
        if len(self.buffer) < self.min_buffer:
            return None

        s, a, r, s2, term = self.buffer.sample(self.batch_size)

        # Q(s, a) for the actions that were actually taken
        q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # TD target: r + gamma * max_a' Q_target(s', a'); no bootstrap if the pole fell
        with torch.no_grad():
            max_next_q = self.target_net(s2).max(dim=1).values
            target = r + self.gamma * (1.0 - term) * max_next_q

        loss = F.smooth_l1_loss(q, target)  # Huber loss: robust to large TD errors
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Hard update of the target network every `target_update_freq` gradient steps
        self.n_updates += 1
        if self.n_updates % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
