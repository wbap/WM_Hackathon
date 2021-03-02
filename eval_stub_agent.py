#!/usr/bin/env python

#
# Trains a simple agent on an environment
# python simple_agent.py Env-vN
#

import os
import sys
import math
import json
import collections
from statistics import mean

import ray
import ray.rllib.agents.a3c as a3c
import ray.tune as tune
from agent.agent import Agent
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch

from utils.writer_singleton import WriterSingleton

torch, nn = try_import_torch()

"""
Create a simple RL agent using a stub Agent.
The environment can be chosen. Both environment and agent are configurable.
"""


def stub_env_creator(task_env_type, task_env_config_file, stub_env_config_file):
  """Custom functor to create custom Gym environments."""
  from gym_game.envs import AgentEnv
  return AgentEnv(task_env_type, task_env_config_file, stub_env_config_file)  # Instantiate with config fil


if len(sys.argv) < 4:
    print('Usage: python simple_agent.py ENV_NAME ENV_CONFIG_FILE STUB_ENV_CONFIG_FILE AGENT_CONFIG_FILE')
    sys.exit(-1)

meta_env_type = 'stub-v0'
task_env_type = sys.argv[1]
print('Task Gym[PyGame] environment:', task_env_type)
task_env_config_file = sys.argv[2]
print('Task Env config file:', task_env_config_file)
stub_env_config_file = sys.argv[3]
print('Stub Env config file:', stub_env_config_file)
model_config_file = sys.argv[4]
print('Agent config file:', model_config_file)

# Try to instantiate the environment
env = stub_env_creator(task_env_type, task_env_config_file, stub_env_config_file)  #gym.make(env_name, config_file=env_config_file)
tune.register_env(meta_env_type, lambda config: stub_env_creator(task_env_type, task_env_config_file, stub_env_config_file))

# Check action space of the environment
if not hasattr(env.action_space, 'n'):
    raise Exception('Only supports discrete action spaces')
ACTIONS = env.action_space.n
print("ACTIONS={}".format(ACTIONS))

# Some general preparations...
render_mode = 'rgb_array'
ray.shutdown()
ray.init(ignore_reinit_error=True)

# Build agent config
agent_config = {}
agent_config["log_level"] = "DEBUG"
agent_config["framework"] = "torch"
agent_config["num_workers"] = 1
agent_config["model"] = {}  # This is the "model" for the agent (i.e. Basal-Ganglia) only.

# Override preprocessor and model
model_name = 'agent_model'
preprocessor_name = 'stub_preprocessor'
agent_config["model"]["custom_model"] = model_name
#agent_config["model"]["custom_preprocessor"] = preprocessor_name

# Adjust model hyperparameters to tune
agent_config["model"]["fcnet_activation"] = 'tanh'
agent_config["model"]["fcnet_hiddens"] = [128, 128]
agent_config["model"]["max_seq_len"] = 50  # TODO Make this a file param. Not enough probably.
agent_config["model"]["framestack"] = False  # default: True

# We're meant to be able to use this key for a custom config dic, but if we set any values, it causes a crash
# https://github.com/ray-project/ray/blob/master/rllib/models/catalog.py
agent_config["model"]["custom_model_config"] = {}

# Override from model config file:
if model_config_file is not None:
  with open(model_config_file) as json_file:
    delta_config = json.load(json_file)

    # Override model config
    model_delta_config = delta_config['model']
    for key, value in model_delta_config.items():
      print('Agent.model config: ', key, ' --> ', value)
      agent_config["model"][key] = value

    # Override agent config
    agent_delta_config = delta_config['agent']
    for key, value in agent_delta_config.items():
      print('Agent config: ', key, ' --> ', value)
      agent_config[key] = value

    # Load parameters that control the training regime
    training_config = delta_config['training']
    evaluation_steps = training_config['evaluation_steps']
    checkpoint_path = training_config['checkpoint_path']

# Register the custom items
ModelCatalog.register_custom_model(model_name, Agent)

print('Agent config:\n', agent_config)
# agent_config['gamma'] = 0.0
agent = a3c.A3CTrainer(agent_config, env=meta_env_type)  # Note use of custom Env creator fn

agent.restore(checkpoint_path)

# Use this line uncommented to see the whole config and all options
# print('\n\n\nPOLICY CONFIG',agent.get_policy().config,"\n\n\n")

# Evaluate the model

def find_json_value(key_path, json, delimiter='.'):
  paths = key_path.split(delimiter)
  data = json
  for i in range(0, len(paths)):
    data = data[paths[i]]
  return data


def get_result(result_step, result_key):
  value = find_json_value(result_key, result_step)
  if not math.isnan(value):
    return value


result_writer_keys = [
  'info.learner.policy_entropy',
  'info.learner.policy_loss',
  'info.learner.vf_loss',
  'info.num_steps_sampled',
  'info.num_steps_trained',
  'episode_reward_min',
  'episode_reward_mean',
  'episode_reward_max'
]

scores = {}

print('Evaluation')
agent.get_policy().config['explore'] = False  # Inference mode

episode_reward = 0
done = False
obs = env.reset()
while not done:
  action = agent.compute_action(obs)
  obs, reward, done, info = env.step(action)
  episode_reward += reward

print('episode reward =', episode_reward)

# Finish
print('Shutting down...')
ray.shutdown()
