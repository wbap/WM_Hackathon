#!/usr/bin/env python

#
# Trains agent on an environment
# python simple_agent.py Env-vN
#

import math
import json
import shutil
import sys
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
Create a simple RL agent using an Agent. 
The environment can be chosen. Both environment and agent are configurable.
"""


def env_creator(task_env_type, task_env_config_file, env_config_file):
  """Custom functor to create custom Gym environments."""
  from gym_game.envs import AgentEnv
  return AgentEnv(task_env_type, task_env_config_file, env_config_file)  # Instantiate with config fil


if len(sys.argv) < 4:
    print('Usage: python simple_agent.py ENV_NAME ENV_CONFIG_FILE ENV_CONFIG_FILE AGENT_CONFIG_FILE')
    sys.exit(-1)

meta_env_type = 'stub-v0'
task_env_type = sys.argv[1]
print('Task Gym[PyGame] environment:', task_env_type)
task_env_config_file = sys.argv[2]
print('Task Env config file:', task_env_config_file)
env_config_file = sys.argv[3]
print('Env config file:', env_config_file)
model_config_file = sys.argv[4]
print('Agent config file:', model_config_file)

# Try to instantiate the environment
env = env_creator(task_env_type, task_env_config_file, env_config_file)  #gym.make(env_name, config_file=env_config_file)
tune.register_env(meta_env_type, lambda config: env_creator(task_env_type, task_env_config_file, env_config_file))

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
preprocessor_name = 'obs_preprocessor'
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
    training_steps = training_config['training_steps']
    training_epochs = training_config['training_epochs']
    evaluation_steps = training_config['evaluation_steps']
    evaluation_interval = training_config['evaluation_interval']
    checkpoint_interval = training_config['checkpoint_interval']

# Register the custom items
ModelCatalog.register_custom_model(model_name, Agent)

print('Agent config:\n', agent_config)
#agent_config['gamma'] = 0.0
agent = a3c.A3CTrainer(agent_config, env=meta_env_type)  # Note use of custom Env creator fn

# Use this line uncommented to see the whole config and all options
#print('\n\n\nPOLICY CONFIG',agent.get_policy().config,"\n\n\n")


# Train the model
writer = WriterSingleton.get_writer()
results_min = collections.deque()
results_mean = collections.deque()
results_max = collections.deque()
results_window_size = 100

checkpoint_dir = writer.get_logdir()

status_message = "{:3d}: reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f}"
file_name = 'None'


def update_results(result_step, results_list, result_key):
  value = result_step[result_key]
  if not math.isnan(value):
    results_list.append(value)
  while len(results_list) > results_window_size:
    results_list.popleft()
  # calculate stats:
  if len(results_list) < 1:
    return 0.0  # can't mean if there's a nan
  mean_value = mean(results_list)
  #print('list:', results)
  #print('list mean:', mean_value)
  return mean_value


def find_json_value(key_path, json, delimiter='.'):
  paths = key_path.split(delimiter)
  data = json
  for i in range(0, len(paths)):
    data = data[paths[i]]
  return data


def update_writer(result_step, result_key, writer, writer_key, step):
  value = find_json_value(result_key, result_step)
  if not math.isnan(value):
    writer.add_scalar(writer_key, value, step)


result_writer_keys = [
  'info.learner.policy_entropy',
  'info.learner.policy_loss',
  'info.learner.vf_loss',
  'info.num_steps_sampled',
  'info.num_steps_trained',
  'episode_reward_min',
  'episode_reward_mean',
  'episode_reward_max']

# TRAINING STARTS
evaluation_epoch = 0
for training_epoch in range(training_epochs):  # number of epochs for all training
  # Train for many steps
  print('Training Epoch ~~~~~~~~~> ', training_epoch)

  # https://github.com/ray-project/ray/issues/8189 - inference mode
  agent.get_policy().config['explore'] = True  # Revert to training
  for training_step in range(training_steps):  # steps in an epoch
    result = agent.train()  # Runs a whole Episode, which includes several tasks and a tutoring phase
    #print('>>> Result: \n\n', result)  # Use this find examine additional keys in the results for plotting

    # Calculate moving averages for console reporting
    mean_min  = update_results(result, results_min, "episode_reward_min")
    mean_mean = update_results(result, results_mean, "episode_reward_mean")
    mean_max  = update_results(result, results_max, "episode_reward_max")
    mean_len = len(results_mean)

    # Update tensorboard plots
    global_step = training_epoch * training_steps + training_step
    for result_key in result_writer_keys:
      writer_key = 'Train/' + result_key
      update_writer(result, result_key, writer, writer_key, global_step)
    writer.flush()

    print(status_message.format(
      global_step,
      mean_min,
      mean_mean,
      mean_max,
      mean_len
     ))

  # Periodically save checkpoints
  if checkpoint_interval > 0:  # Optionally save checkpoint every n epochs of training
    if (training_epoch % checkpoint_interval) == 0:
      file_name = agent.save(checkpoint_dir)

  # Always save the final checkpoint regardless of interval
  if training_epoch == (training_epochs - 1):
    file_name = agent.save(checkpoint_dir)

  # Periodically evaluate
  if (training_epoch % evaluation_interval) == 0:
    print('Evaluation Epoch ~~~~~~~~~> ', evaluation_epoch)
    agent.get_policy().config['explore'] = False  # Inference mode
    for evaluation_step in range(evaluation_steps):  # steps in an epoch
      result = agent.train()
      # TODO use compute_action
      global_step = evaluation_epoch * evaluation_steps + evaluation_step
      for result_key in result_writer_keys:
        writer_key = 'Eval/' + result_key
        update_writer(result, result_key, writer, writer_key, global_step)
      writer.flush()

    evaluation_epoch = evaluation_epoch + 1

# Finish
print('Shutting down...')
ray.shutdown()
