#!/usr/bin/env python
import sys, gym, time
import json

#
# (Pre)Trains a simple agent modules in an unsupervised/self-supervised manner on prerecorded samples from an environment.
# E.g. 
# python pretrain.py dm2s-v0 ../games/dm2s/DM2S.par data.pickle simple_model.json
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

from gym_game.envs.pygame_dataset import PyGameDataset

if len(sys.argv) < 4:
    print('Usage: python simple_agent.py ENV_NAME ENV_CONFIG_FILE DATASET_FILE MODEL_CONFIG_FILE')
    sys.exit(-1)

env_name = sys.argv[1]
print('Making Gym[PyGame] environment:', env_name)
env_config_file = sys.argv[2]
print('Env config file:', env_config_file)
dataset_file = sys.argv[3]
print('Dataset file:', dataset_file)
model_config_file = sys.argv[4]
print('Model config file:', model_config_file)


# Check environment
env = gym.make(env_name, config_file=env_config_file)
if not hasattr(env.action_space, 'n'):
    raise Exception('Simple agent only supports discrete action spaces')
ACTIONS = env.action_space.n
print("ACTIONS={}".format(ACTIONS))


# Check Ray
ray.shutdown()
ray.init(ignore_reinit_error=True)
CHECKPOINT_ROOT = 'tmp/simple'
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)


# Configs
config= {}
config["log_level"] = "DEBUG"
config["framework"] = "torch"
config["num_workers"] = 1
config["model"] = {}
model_name = 'flat_custom_model'
config["model"]["custom_model"] = model_name

# Adjust model hyperparameters to tune
config["model"]["fcnet_activation"] = 'tanh'
config["model"]["fcnet_hiddens"] = [128, 128]
config["model"]["max_seq_len"] = 50
config["model"]["framestack"] = False  # default: True

# Override from model config file:
if model_config_file is not None:
    with open(model_config_file) as json_file:
        model_config = json.load(json_file)
        for key, value in model_config.items():
            #print('Override key:', key, 'value:', value)
            config["model"][key] = value
print('Final complete config: ', config)


# Build and reload pregenerated data
dataset = PyGameDataset()
ok = dataset.read(dataset_file)
if ok:
    print('Read pregenerated dataset OK.')

# Custom env creator
def env_creator(env_name, env_config_file):
    """Custom functor to create custom Gym environments."""
    if env_name == 'simple-v0':
        from gym_game.envs import SimpleEnv as env
    else:
        raise NotImplementedError
    return env(env_config_file)  # Instantiate with config file
tune.register_env(env_name, lambda config: env_creator(env_name, env_config_file))



# Build entire model
class TorchCustomModel(TorchModelV2, nn.Module):
    """PyTorch custom model that flattens the input to 1d and delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # Reshape obs to vector and convert to float
        volume = np.prod(obs_space.shape)
        space = np.zeros(volume)
        flat_observation_space = spaces.Box(low=0, high=255, shape=space.shape, dtype=np.float32)

        # TODO: Transform to output of any other PyTorch and pass new shape to model.

        # Create default model
        TorchModelV2.__init__(self, flat_observation_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.torch_sub_model = TorchFC(flat_observation_space, action_space, num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        # flatten
        obs_4d = input_dict["obs"].float()
        volume = np.prod(obs_4d.shape[1:])  # calculate volume as vector excl. batch dim
        obs_3d_shape = [obs_4d.shape[0], volume]  # [batch size, volume]
        obs_3d = np.reshape(obs_4d, obs_3d_shape)
        input_dict["obs"] = obs_3d

        # TODO: forward() any other PyTorch modules here, pass result to RL algo

        # Defer to default FC model
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])

# Register the model
ModelCatalog.register_custom_model(model_name, TorchCustomModel)

#agent = a3c.A3CTrainer(config, env=env_creator(env_name, env_config_file))  # Note use of custom Env creator fn
#agent = a3c.A3CTrainer(config, env=env_name)  # Note use of custom Env creator fn

# Train the model
status_message = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"
agent_steps = 250
#for n in range(agent_steps):

# Finish
print('Shutting down...')
