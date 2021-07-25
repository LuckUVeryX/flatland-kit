import random

import numpy as np
import torch

from our_policies.policy import LearningPolicy, Policy
from our_policies.replay_buffer import ReplayBuffer


class RandomPolicy(LearningPolicy):
    def __init__(self, state_size, action_size, in_parameters, evaluation_mode=False):
        print(">> RandomPolicy")
        super(Policy, self).__init__()

        self.random_parameters = in_parameters
        self.action_size = action_size

        # Device
        if self.random_parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("🐇 Using GPU")
        else:
            self.device = torch.device("cpu")
            print("🐢 Using CPU")

        self.loss = 0.0

        self.memory = ReplayBuffer(action_size, 1, 1, self.device)

    def step(self, handle, state, action, reward, next_state, done):
        pass

    def act(self, handle, state, eps=0.):
        return random.choice(np.arange(self.action_size))

    def save(self, filename):
        pass

    def load(self, filename):
        pass
