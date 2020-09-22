#!/usr/bin/env python
import sys, gym, time
import json

#
# Generates a large dataset of samples from an environment with a random policy
#
import numpy as np
from gym import error, spaces, utils
import gym_game
import pygame

import ray
import ray.tune as tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
import shutil
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()

from gym_game.envs.pygame_pretrain_policy import PyGamePretrainPolicy
from gym_game.envs.pygame_dataset import PyGameDataset

if len(sys.argv) < 4:
    print('Usage: python simple_agent.py ENV_NAME ENV_CONFIG_FILE NUM_SAMPLES DATA_DIR')
    sys.exit(-1)

env_name = sys.argv[1]
print('Making Gym[PyGame] environment:', env_name)
env_config_file = sys.argv[2]
print('Env config file:', env_config_file)
num_samples = int(sys.argv[3])
print('Number of samples:', num_samples)
file_name = sys.argv[4]
print('Data directory:', file_name)

# Custom env creator
def env_creator(env_name, env_config_file):
    """Custom functor to create custom Gym environments."""
    if env_name == 'simple-v0':
        from gym_game.envs import SimpleEnv as env
    else:
        raise NotImplementedError
    return env(env_config_file)  # Instantiate with config file

tune.register_env(env_name, lambda config: env_creator(env_name, env_config_file))

env = gym.make(env_name, config_file=env_config_file)
policy = PyGamePretrainPolicy(env.action_space)
dataset = PyGameDataset()
print('Generating samples...')
dataset.generate(num_samples, env, policy)
print('Writing samples...')
dataset.write(file_name)

# Uncomment to test recovery of the data
#dataset.read(file_name)