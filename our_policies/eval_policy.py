import os
from policy import Policy
import random
import sys
from argparse import ArgumentParser, Namespace
from collections import deque
from datetime import datetime
from pathlib import Path
from pprint import pprint
import time

import numpy as np
import psutil
from flatland.envs.malfunction_generators import (MalfunctionParameters,
												  malfunction_from_params)
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from torch.utils.tensorboard import SummaryWriter
from utils.agent_action_config import (get_action_size,
									   get_flatland_full_action_size,
									   map_action, map_action_policy,
									   map_actions, set_action_size_full,
									   set_action_size_reduced)
from utils.fast_tree_obs import FastTreeObs
from utils.observation_utils import normalize_observation
from utils.timer import Timer

# ! Import our policies
from random_policy import RandomPolicy
from go_forward_policy import GoForwardPolicy
from dddqn import DDDQNPolicy

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))


try:
	import wandb

	wandb.init(sync_tensorboard=True)
except ImportError:
	print("Install wandb to log to Weights & Biases")

def create_rail_env(env_params, tree_observation):
	n_agents = env_params.n_agents
	x_dim = env_params.x_dim
	y_dim = env_params.y_dim
	n_cities = env_params.n_cities
	max_rails_between_cities = env_params.max_rails_between_cities
	max_rails_in_city = env_params.max_rails_in_city
	seed = env_params.seed

	# Break agents from time to time
	malfunction_parameters = MalfunctionParameters(
		malfunction_rate=env_params.malfunction_rate,
		min_duration=20,
		max_duration=50
	)

	return RailEnv(
		width=x_dim, height=y_dim,
		rail_generator=sparse_rail_generator(
			max_num_cities=n_cities,
			grid_mode=False,
			max_rails_between_cities=max_rails_between_cities,
			max_rails_in_city=max_rails_in_city
		),
		schedule_generator=sparse_schedule_generator(),
		number_of_agents=n_agents,
		malfunction_generator_and_process_data=malfunction_from_params(
			malfunction_parameters),
		obs_builder_object=tree_observation,
		random_seed=seed
	)

def eval_policy(env, tree_observation, policy, train_params, obs_params):
	n_eval_episodes = train_params.n_evaluation_episodes
	max_steps = 50
	# max_steps = env._max_episode_steps
	tree_depth = obs_params.observation_tree_depth
	observation_radius = obs_params.observation_radius

	print(max_steps)
	action_dict = dict()
	scores = []
	completions = []
	nb_steps = []
	prev_obs = [None] * env.get_num_agents()

	for episode_idx in range(n_eval_episodes):
		agent_obs = [None] * env.get_num_agents()
		score = 0.0

		obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
		policy.reset(env)
		final_step = 0

		# Build initial obs
		for agent in env.get_agent_handles():
			if obs[agent] is not None:
				agent_obs[agent] = obs[agent]
				prev_obs[agent] = obs[agent]

		env_renderer = RenderTool(eval_env, gl="PGL")
		env_renderer.set_new_rail()
		policy.start_episode(train=False)
		for step in range(max_steps - 1):
			policy.start_step(train=False)
			# print(sum(x is None for x in prev_obs))
			for agent in env.get_agent_handles():
				if obs[agent] is not None:
					prev_obs[agent] = obs[agent]
					agent_obs[agent] = tree_observation.get_normalized_observation(obs[agent], tree_depth=tree_depth, observation_radius=observation_radius)
					
				if obs[agent] is None:
					print(f"{agent} has NONE %%%%%%%%%%%%%%")
					agent_obs[agent] = tree_observation.get_normalized_observation(prev_obs[agent], tree_depth=tree_depth, observation_radius=observation_radius)

				action = 0
				if info['action_required'][agent]:
					action = policy.act(agent, agent_obs[agent], eps=0.0)
					
				action_dict.update({agent: action})
			policy.end_step(train=False)
			obs, all_rewards, done, info = env.step(map_action(action_dict))
			# print(action_dict)
			env_renderer.render_env(
								show=True,
								frames=False,
								show_observations=False,
								show_predictions=True
                )

			# time.sleep(2)
			for agent in env.get_agent_handles():
				score += all_rewards[agent]

			final_step = step

			if done['__all__']:
				break
		policy.end_episode(train=False)
		normalized_score = score / (max_steps * env.get_num_agents())
		scores.append(normalized_score)

		tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
		completion = tasks_finished / max(1, env.get_num_agents())
		completions.append(completion)

		nb_steps.append(final_step)
		env_renderer.close_window()
	print(" ✅ Eval: score {:.3f} done {:.1f}%".format(
		np.mean(scores), np.mean(completions) * 100.0))

	return scores, completions, nb_steps
	
 
# Observation parameters
obs_params = {
	"observation_tree_depth": 2,
	"observation_radius": 10,
	"observation_max_path_depth": 30
}
observation_tree_depth = obs_params["observation_tree_depth"]
observation_radius = obs_params["observation_radius"]
observation_max_path_depth = obs_params["observation_max_path_depth"]

# Evaluation Environment Parameters
eval_env_params = {'malfunction_rate': 0,
 'max_rails_between_cities': 2,
 'max_rails_in_city': 2,
 'n_agents': 5,
 'n_cities': 3,
 'seed': 0,
 'x_dim': 25,
 'y_dim': 25}

# Train Parameters
train_params = {'action_size': 'full',
 'batch_size': 128,
 'buffer_min_size': 0,
 'buffer_size': 32000,
 'checkpoint_interval': 100,
 'eps_decay': 0.9975,
 'eps_end': 0.01,
 'eps_start': 1.0,
 'evaluation_env_config': 1,
 'gamma': 0.97,
 'hidden_size': 128,
 'learning_rate': 5e-05,
 'load_policy': '',
 'max_depth': 2,
 'n_agent_fixed': False,
 'n_episodes': 1000,
 'n_evaluation_episodes': 10,
 'num_threads': 4,
 'policy': 'dddqn',
 'render': False,
 'restore_replay_buffer': '',
 'save_replay_buffer': False,
 'tau': 0.0005,
 'training_env_config': 1,
 'update_every': 10,
 'use_fast_tree_observation': False,
 'use_gpu': True}

scores_list = []
completions_list = []
nb_steps_list = []


predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
print("\nUsing standard TreeObs")

def check_is_observation_valid(observation):
	return observation

def get_normalized_observation(observation, tree_depth: int, observation_radius=0):
	return normalize_observation(observation, tree_depth, observation_radius)

tree_observation = TreeObsForRailEnv(
	max_depth=observation_tree_depth, predictor=predictor)
tree_observation.check_is_observation_valid = check_is_observation_valid
tree_observation.get_normalized_observation = get_normalized_observation

eval_env = create_rail_env(Namespace(**eval_env_params), tree_observation)
eval_env.reset(regenerate_schedule=True, regenerate_rail=True)


n_features_per_node = eval_env.obs_builder.observation_dim
n_nodes = sum([np.power(4, i)
				for i in range(observation_tree_depth + 1)])
state_size = n_features_per_node * n_nodes
policy = DDDQNPolicy(state_size, 5, Namespace(**train_params), evaluation_mode=True)
'''################## CHANGE YOUR FILE PATH HERE ######################'''
policy.load("D:\\SUTD\\Term 8 ESD ISTD\\50.021 Artificial Intelligence\\Project\\flatland-kit\\martz_runs\\defaultParams-fixedEval-5000\\210801093939-5000.pth")
''' Make sure that you don't include the .target and .local extensions, check load method under dddqn policy for details '''

scores, completions, nb_steps_eval = eval_policy(eval_env,
												tree_observation,
												policy,
												Namespace(**train_params),
												Namespace(**obs_params))
scores_list.append(scores)
completions_list.append(completions)
nb_steps_list.append(nb_steps_eval)

print(scores_list)
print(completions_list)
print(nb_steps_list)