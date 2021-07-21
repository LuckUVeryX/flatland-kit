import random

import numpy as np

from reinforcement_learning.policy import Policy


class RandomPolicy(Policy):
    def __init__(self):
        self.action_size = 5

    def step(self, state, action, reward, next_state, done):
        pass

    def act(self, state, eps=0.):
        return random.choice(np.arange(self.action_size))

    def save(self, filename):
        pass

    def load(self, filename):
        pass
