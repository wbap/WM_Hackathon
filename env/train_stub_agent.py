#!/usr/bin/env python

#
# Trains a simple agent on an environment
# python simple_agent.py Env-vN
#

import json
import shutil
import sys

import gym
import ray
import ray.rllib.agents.a3c as a3c
import ray.tune as tune
from agent.stub_agent import StubAgent
from agent.stub_agent import StubPreprocessor
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

def stub_env_creator(task_env_type, task_env_config_file, stub_env_config_file):
  """Custom functor to create custom Gym environments."""
  from gym_game.envs import StubAgentEnv
  return StubAgentEnv(task_env_type, task_env_config_file, stub_env_config_file)  # Instantiate with config fil

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
CHECKPOINT_ROOT = 'tmp/simple'
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

# Build agent config
agent_config = {}
agent_config["log_level"] = "DEBUG"
agent_config["framework"] = "torch"
agent_config["num_workers"] = 1
agent_config["model"] = {}  # This is the "model" for the agent (i.e. Basal-Ganglia) only.

# Override preprocessor and model
model_name = 'stub_agent_model'
preprocessor_name = 'stub_preprocessor'
agent_config["model"]["custom_model"] = model_name
#agent_config["model"]["custom_preprocessor"] = preprocessor_name

# Adjust model hyperparameters to tune
agent_config["model"]["fcnet_activation"] = 'tanh'
agent_config["model"]["fcnet_hiddens"] = [128, 128]
agent_config["model"]["max_seq_len"] = 50
agent_config["model"]["framestack"] = False  # default: True

# We're meant to be able to use this key for a custom config dic, but if we set any values, it causes a crash
# https://github.com/ray-project/ray/blob/master/rllib/models/catalog.py
agent_config["model"]["custom_model_config"] = {}

# Override from model config file:
if model_config_file is not None:
  with open(model_config_file) as json_file:
    delta_config = json.load(json_file)
    model_config = delta_config['model']
    for key, value in model_config.items():
      print('Agent model config: ', key, ' --> ', value)
      agent_config["model"][key] = value

    training_config = delta_config['training']
    training_steps = training_config['steps']
    checkpoint_interval = training_config['checkpoint_interval']

# Register the custom items
ModelCatalog.register_custom_model(model_name, StubAgent)

print('Agent config:\n', agent_config)
agent = a3c.A3CTrainer(agent_config, env=meta_env_type)  # Note use of custom Env creator fn

# Train the model
status_message = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"
file_name = 'n/a'
for n in range(training_steps):
  result = agent.train()
  if checkpoint_interval > 0:
    if (n % checkpoint_interval) == 0:
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
