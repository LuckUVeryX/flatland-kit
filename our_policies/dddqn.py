import copy 
import os
import pickle 
import random

import numpy as np 
import torch 
import torch.nn.functional as F 
import torch.optim as optim
import torch.nn as nn
from our_policies.policy import Policy, LearningPolicy
from our_policies.replay_buffer import ReplayBuffer

class DuelingQNetwork(nn.Module): 
    def __init__(self, state_size, action_size, hidsize1=128, hidsize2=128): 
        super(DuelingQNetwork, self).__init__()

        self.fc1_val = nn.Linear(state_size, hidsize1)
        self.fc2_val = nn.Linear(hidsize1, hidsize2)
        self.fc4_val = nn.Linear(hidsize2, 1)

        self.fc1_adv = nn.Linear(state_size, hidsize1)
        self.fc2_adv = nn.Linear(hidsize1, hidsize2)
        self.fc4_adv = nn.Linear(hidsize2, action_size)

    def forward(self, x): 
        val = F.relu(self.fc1_val(x))
        val = F.relu(self.fc2_val(val))
        val = self.fc4_val(val)

        adv = F.relu(self.fc1_adv(x))
        adv = F.relu(self.fc2_adv(adv))
        adv = self.fc4_adv(adv)

        return val + adv - adv.mean()

class DDDQNPolicy(LearningPolicy):
    """Dueling Double DQN policy"""

    def __init__(self, state_size, action_size, in_parameters, evaluation_mode=False):
        print(">> DDDQNPolicy")
        super(Policy, self).__init__()

        self.ddqn_parameters = in_parameters
        self.evaluation_mode = evaluation_mode

        self.state_size = state_size
        self.action_size = action_size
        self.double_dqn = True
        self.hidsize = 256

        if not evaluation_mode:
            self.hidsize = self.ddqn_parameters.hidden_size
            self.buffer_size = self.ddqn_parameters.buffer_size
            self.batch_size = self.ddqn_parameters.batch_size
            self.update_every = self.ddqn_parameters.update_every
            self.learning_rate = self.ddqn_parameters.learning_rate
            self.tau = self.ddqn_parameters.tau
            self.gamma = self.ddqn_parameters.gamma
            self.buffer_min_size = self.ddqn_parameters.buffer_min_size

            # Device
        if self.ddqn_parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            # print("???? Using GPU")
        else:
            self.device = torch.device("cpu")
            # print("???? Using CPU")

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size,
                                              action_size,
                                              hidsize1=self.hidsize,
                                              hidsize2=self.hidsize).to(self.device)

        if not evaluation_mode:
            self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
            self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)
            self.t_step = 0
            self.loss = 0.0
        else:
            self.memory = ReplayBuffer(action_size, 1, 1, self.device)
            self.loss = 0.0

    def act(self, handle, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() >= eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, handle, state, action, reward, next_state, done):
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
                self._learn()

    def _learn(self):
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones, _ = experiences

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        if self.double_dqn:
            # Double DQN
            q_best_action = self.qnetwork_local(next_states).max(1)[1]
            q_targets_next = self.qnetwork_target(next_states).gather(1, q_best_action.unsqueeze(-1))
        else:
            # DQN
            q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(-1)

        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Compute loss
        self.loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def _soft_update(self, local_model, target_model, tau):
        # Soft update model parameters.
        # ??_target = ??*??_local + (1 - ??)*??_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.qnetwork_local.state_dict(), filename + ".local")
        torch.save(self.qnetwork_target.state_dict(), filename + ".target")

    def load(self, filename):
        try:
            if os.path.exists(filename + ".local") and os.path.exists(filename + ".target"):
                self.qnetwork_local.load_state_dict(torch.load(filename + ".local", map_location=self.device))
                print("qnetwork_local loaded ('{}')".format(filename + ".local"))
                if not self.evaluation_mode:
                    self.qnetwork_target.load_state_dict(torch.load(filename + ".target", map_location=self.device))
                    print("qnetwork_target loaded ('{}' )".format(filename + ".target"))
            else:
                print(">> Checkpoint not found, using untrained policy! ('{}', '{}')".format(filename + ".local",
                                                                                             filename + ".target"))
        except Exception as exc:
            print(exc)
            print("Couldn't load policy from, using untrained policy! ('{}', '{}')".format(filename + ".local",
                                                                                           filename + ".target"))

    def save_replay_buffer(self, filename):
        memory = self.memory.memory
        with open(filename, 'wb') as f:
            pickle.dump(list(memory)[-500000:], f)

    def load_replay_buffer(self, filename):
        with open(filename, 'rb') as f:
            self.memory.memory = pickle.load(f)

    def test(self):
        self.act(0, np.array([[0] * self.state_size]))
        self._learn()

    def clone(self):
        me = DDDQNPolicy(self.state_size, self.action_size, self.ddqn_parameters, evaluation_mode=True)
        me.qnetwork_target = copy.deepcopy(self.qnetwork_local)
        me.qnetwork_target = copy.deepcopy(self.qnetwork_target)
        return me