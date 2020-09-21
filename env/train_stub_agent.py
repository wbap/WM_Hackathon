#!/usr/bin/env python

#
# Trains a simple agent on an environment
# python simple_agent.py Env-vN
#

import gym
import json
import shutil
import sys

import ray
import ray.rllib.agents.a3c as a3c
import ray.tune as tune
from agent.stub_agent import StubAgent
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

"""
Create a simple RL agent using StubAgent. 
The environment can be chosen. Both environment and agent are configurable.

Usage: python simple_agent.py ENV_NAME ENV_CONFIG_FILE MODEL_CONFIG_FILE
e.g.
  python simple_agent.py dm2s-v0 ../games/dm2s/DM2S.par simple_agent_model.json
  --> the StubAgent plays the dm2s game

  python simple_agent.py simple-v0 simple_env_machine.json simple_agent_model.json
  --> the StubAgent (conf simple_agent_model.json) plays the simple env (config simple_env_machine.json)  
"""

if len(sys.argv) < 4:
    print('Usage: python simple_agent.py ENV_NAME ENV_CONFIG_FILE MODEL_CONFIG_FILE')
    sys.exit(-1)

env_name = sys.argv[1]
print('Making Gym[PyGame] environment:', env_name)
env_config_file = sys.argv[2]
print('Env config file:', env_config_file)
env = gym.make(env_name, config_file=env_config_file)

model_config_file = sys.argv[3]
print('Model config file:', model_config_file)

if not hasattr(env.action_space, 'n'):
    raise Exception('Simple agent only supports discrete action spaces')
ACTIONS = env.action_space.n

print("ACTIONS={}".format(ACTIONS))

render_mode = 'rgb_array'

ray.shutdown()
ray.init(ignore_reinit_error=True)

CHECKPOINT_ROOT = 'tmp/simple'
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

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

# We're meant to be able to use this key for a custom config dic, but if we set any values, it causes a crash
# https://github.com/ray-project/ray/blob/master/rllib/models/catalog.py
config["model"]["custom_model_config"] = {}

# Override from model config file:
if model_config_file is not None:
    with open(model_config_file) as json_file:
        model_config = json.load(json_file)
        for key, value in model_config.items():
            #print('Override key:', key, 'value:', value)
            config["model"][key] = value
print('Final complete config: ', config)


def env_creator(env_name, env_config_file):
    """Custom functor to create custom Gym environments."""
    if env_name == 'simple-v0':
      from gym_game.envs import SimpleEnv as env
    elif env_name == 'dm2s-v0':
      from gym_game.envs import Dm2sEnv as env
    else:
      raise NotImplementedError
    return env(env_config_file)  # Instantiate with config file


# Register the model
# https://github.com/ray-project/ray/issues/9040
ModelCatalog.register_custom_model(model_name, StubAgent)
# https://stackoverflow.com/questions/58551029/rllib-use-custom-registered-environments
#tune.register_env(env_name, lambda: config, env_creator(env_name, env_config_file))
tune.register_env(env_name, lambda config: env_creator(env_name, env_config_file))
#agent = a3c.A3CTrainer(config, env=env_creator(env_name, env_config_file))  # Note use of custom Env creator fn
agent = a3c.A3CTrainer(config, env=env_name)  # Note use of custom Env creator fn

# Train the model
status_message = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"
agent_steps = 250
for n in range(agent_steps):
  result = agent.train()
  file_name = agent.save(CHECKPOINT_ROOT)
  print(status_message.format(
    n + 1,
    result["episode_reward_min"],
    result["episode_reward_mean"],
    result["episode_reward_max"],
    result["episode_len_mean"],
    file_name
   ))

# Finish
print('Shutting down...')
ray.shutdown()
