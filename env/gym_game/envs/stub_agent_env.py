import math
import json
from collections import deque

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from abc import ABC, abstractmethod
import numpy as np
import pygame as pygame

from gym_game.stubs.positional_encoder import PositionalEncoder
from gym_game.stubs.posterior_cortex import PosteriorCortex
from gym_game.stubs.image_utils import *

from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()
from timeit import default_timer as timer

"""
  Wraps a task-specific environment and implements brain modules that are not trained by Reinforcement Learning.
"""


def prefrontal_cortex(mtl, bg_action):
  pfc_action = bg_action
  print("======> StubAgent: bg_action", bg_action)
  return pfc_action


def superior_colliculus(pfc_action):
  """
    pfc_action: command from PFC. Gaze target in 'action' space
    Return: gaze target in absolute coordinates (pixels in screen space)

    Currently, this is a 'pass-through" component.
    In the future, one may want to change the implementation e.g. progressively move toward the target
  """

  # absolute coordinates(pixels in screen space)

  sc_action = pfc_action
  print("======> StubAgentEnv: agent_action", sc_action)

  return sc_action


def sc_2_env(sc_action):
  return sc_action


class StubAgentEnv(gym.Env):

  # Streams
  OBS_FOVEA = 'fovea'
  OBS_PERIPHERAL = 'peripheral'
  OBS_POSITIONAL_ENCODING = 'gaze'
  NUM_OBS = 2

  @staticmethod
  def get_default_config():
    pe_config = PositionalEncoder.get_default_config()
    cortex_f_config = PosteriorCortex.get_default_config()
    cortex_p_config = PosteriorCortex.get_default_config()
    agent_config = {
      'obs_keys': {
        'visual': [StubAgentEnv.OBS_FOVEA, StubAgentEnv.OBS_PERIPHERAL]
      },
      StubAgentEnv.OBS_FOVEA: cortex_f_config,
      StubAgentEnv.OBS_PERIPHERAL: cortex_p_config,
      StubAgentEnv.OBS_POSITIONAL_ENCODING: pe_config
    }
    return agent_config

  @staticmethod
  def update_config(default_config, delta_config):
    """
    Override the config selectively. Return a complete config.
    """
    #updated_config = {**default_config, **delta_config}
    updated_config = dict(mergedicts(default_config, delta_config))
    return updated_config

  def __init__(self, env_type, env_config_file, config_file):
    self.env = gym.make(env_type, config_file=env_config_file)
    #self.env = env_type(env_config_file)
    self.action_space = self.env.action_space
    self.env_observation_space = self.env.observation_space
    self.reward = None

    # Build networks to preprocess the observation space
    default_config = self.get_default_config()  # TODO make this override
    with open(config_file) as json_file:
      delta_config = json.load(json_file)
      self._config = self.update_config(default_config, delta_config)

    print("=======================> CONFIG IS: ", self._config)

    # gather all obs keys
    self._obs_keys = []
    for obs_keys_key in self._config['obs_keys'].keys():
      for obs_key in self._config['obs_keys'][obs_keys_key]:
        self._obs_keys += obs_key

    # build all the components, and add the observation spaces to obs_spaces_dict
    obs_spaces_dict = {}
    self.modules = {}

    # Medial Temporal Lobe
    self.mtl = deque([], self._config["mtl_max_length"])

    # positional encoding
    self._use_pe = "pe" in self._config["obs_keys"]
    if self._use_pe:
      self._build_positional_encoder(obs_spaces_dict)

    # visual processing - create a parietal cortex for fovea and periphery
    self._use_visual = "visual" in self._config["obs_keys"]
    if self._use_visual:
      self._build_visual_stream(obs_spaces_dict)

    # the new observation space dict from the processed streams
    self.observation_space = spaces.Dict(obs_spaces_dict)

  def reset(self):
    #print('>>>>>>>>>>> Stub reset')
    obs = self.env.reset()
    return self.forward(obs)

  # -------------------------------------- Building Regions --------------------------------------
  # ------------ Visual Streams

  @staticmethod
  def create_input_shape_visual(env_observation_space, env_obs_key):
    """ Convert from observation space to PyTorch tensor """
    env_obs = env_observation_space[env_obs_key]
    c_in = env_obs.shape[0]
    h = env_obs.shape[1]
    w = env_obs.shape[2]
    input_shape = [-1, c_in, h, w]
    return input_shape

  @staticmethod
  def create_observation_space_visual(observation_shape):
    """ Convert from observation shape to space"""
    observation_space = spaces.Box(low=-math.inf, high=math.inf, shape=observation_shape, dtype=np.float32)
    return observation_space

  @staticmethod
  def create_observation_shape_visual(network_shape):
    """ Convert from PyTorch tensor to observation shape """
    b = network_shape[0]
    c = network_shape[1]
    h = network_shape[2]
    w = network_shape[3]
    observation_shape = [c, h, w]
    return observation_shape

  def _build_visual_stream(self, obs_spaces_dict):
    for obs_key in self._config["obs_keys"]["visual"]:
      input_shape = self.create_input_shape_visual(self.env_observation_space, obs_key)
      config = self._config[obs_key]
      cortex = PosteriorCortex(obs_key, input_shape, config)
      self.modules[obs_key] = cortex

      output_shape = cortex.get_output_shape()
      obs_shape = self.create_observation_shape_visual(output_shape)
      obs_space = self.create_observation_space_visual(obs_shape)
      obs_spaces_dict.update({obs_key: obs_space})

  # ------------ Positional Encoding
  @staticmethod
  def create_input_shape_pe(env_observation_space, env_obs_key):
    """ Convert from observation space to PyTorch tensor """
    env_obs = env_observation_space[env_obs_key]
    gaze_shape = env_obs.shape[0]   # 1 dimensional list, with length 2 for x and y
    input_shape = [-1, gaze_shape]
    return input_shape

  @staticmethod
  def create_observation_space_pe(observation_shape):
    """ Convert from observation shape to space"""
    observation_space = spaces.Box(low=-math.inf, high=math.inf, shape=observation_shape, dtype=np.float32)
    return observation_space

  @staticmethod
  def create_observation_shape_pe(network_shape):
    """ Convert from PyTorch tensor to observation shape """
    b = network_shape[0]
    xy = network_shape[1]
    observation_shape = [xy]
    return observation_shape

  def _build_positional_encoder(self, obs_spaces_dict):
    obs_key = self.OBS_POSITIONAL_ENCODING
    input_shape = self.create_input_shape_pe(self.env_observation_space, obs_key)
    screen_shape = self.get_screen_shape()
    config = self._config[obs_key]
    pe = PositionalEncoder(obs_key, input_shape, config, max_xy=(screen_shape[0], screen_shape[1]))
    self.modules[obs_key] = pe

    output_shape = pe.get_output_shape()
    obs_shape = self.create_observation_shape_pe(output_shape)
    obs_space = self.create_observation_space_pe(obs_shape)
    obs_spaces_dict.update({obs_key: obs_space})

  # -----------------------------------------------------------------------------------------------

  def forward(self, observation):
    # print('-----------Obs old', observation)

    # process foveal and peripheral parietal cortex
    obs_dict = {}
    if self._use_visual:
      for obs_key in self._config["obs_keys"]["visual"]:
        cortex = self.modules[obs_key]
        input_tensor = self.obs_to_tensor(observation, obs_key)
        encoding_tensor, decoding, target = cortex.forward(input_tensor)
        self.tensor_to_obs(encoding_tensor, obs_dict, obs_key)

    # process positional encoding
    if self._use_pe:
      obs_key = self.OBS_POSITIONAL_ENCODING
      pe = self.modules[obs_key]
      input_tensor = self.obs_to_tensor(observation, obs_key)
      pe_output = pe.forward(input_tensor)

      self.tensor_to_obs(pe_output, obs_dict, obs_key)

    return obs_dict

  def tensor_to_obs(self, output, obs_dict, obs_key):
    #print('output is', output)
    obs = torch.squeeze(output).detach().numpy()  # remove batch dim, detach graph, convert numpy
    #print('!!!!!!!!!!!!!!!!!:',obs_key,' output tensor shape:', obs.shape)
    obs_dict[obs_key] = obs

  def obs_to_tensor(self, observation, obs_key):
    obs = torch.tensor(observation[obs_key])
    obs_b = torch.unsqueeze(obs, 0)  # insert batch dimension 0
    #print('!!!!!!!!!!!!!!!!!:',obs_key,' input tensor shape:', obs_b.shape)
    return obs_b

  def get_config(self):
    """ return a dictionary of params """
    return self.env.get_config()

  def step(self, action):

    debug_observation = False
    debug_timing = False

    start = None
    if debug_timing:
      print('>>>>>>>>>>> Stub step')
      start = timer()

    # compute action for the environment (based on the StubAgent's action on the StubAgentEnv)
    pfc_action = prefrontal_cortex(self.mtl, bg_action=action)
    sc_action = superior_colliculus(pfc_action=pfc_action)
    env_action = sc_2_env(sc_action)

    [obs, self.reward, is_end_state, additional] = self.env.step(env_action)
    tx_obs = self.forward(obs)  # Process the input
    emit = [tx_obs, self.reward, is_end_state, additional]

    # add observations to the MTL
    self.mtl.append(tx_obs)

    # The purpose of this section is to verify that valid observations are emitted.
    if debug_observation:
      print('Tx Obs keys ', tx_obs.keys())
      o = tx_obs['full']
      print('Obs Shape = ', o.shape)
      import hashlib
      m = hashlib.md5()
      m.update(o)
      h = m.hexdigest()
      print(' Hash = ', h)

    if debug_timing:
      end = timer()
      print('Step elapsed time: ', str(end - start))  # Time in seconds, e.g. 5.38091952400282

    return emit

  def get_screen_shape(self):
    return self.env.get_screen_shape()

  def get_random(self):
    """Return the PRNG for this game"""
    return self.env.get_random()

  def seed(self, seed=None):
    return self.env.seed(seed)

  def get_time(self):
    """Returns game time in milliseconds"""
    return self.env.get_time()

  def get_observation(self):
    print('>>>>>>>>>>> Stub get obs')
    obs = self.env.get_observation()
    tx_obs = self.forward(obs)
    return tx_obs

  def render(self, mode='human', close=False):
    return self.env.render(mode, close)


