import math
import json
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from abc import ABC, abstractmethod
import numpy as np
import pygame as pygame

from gym_game.stubs.retina import Retina
from gym_game.stubs.posterior_cortex import PosteriorCortex
# from utils.medial_temporal_lobe import MedialTemporalLobe
# from utils.positional_encoding import PositionalEncoder
# from utils.prefrontal_cortex import PrefrontalCortex
# from utils.superior_colliculus import SuperiorColliculus
from gym_game.stubs.image_utils import *

from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()

"""
  Wraps a task-specific environment and implements brain modules that are not trained by Reinforcement Learning.
"""

class StubAgentEnv(gym.Env):

  # Streams
  OBS_FOVEA = 'fovea'
  OBS_PERIPHERAL = 'peripheral'
  NUM_OBS = 2

  # default config
  # default_config = {
  #   "retina": {
  #     'f_size': 7,
  #     'f_sigma': 2.0,
  #     'f_k': 1.6  # approximates Laplacian of Gaussian
  #   },
  #   "positional_encoding": {},
  #   "vc_fovea": {},
  #   "vc_periphery": {},
  #   "mtl": {},
  #   "sc": {},
  #   "pfc": {}
  # }

  @staticmethod
  def get_default_config():
    cortex_f_config = PosteriorCortex.get_default_config()
    cortex_p_config = PosteriorCortex.get_default_config()
    agent_config = {
      'obs_keys':[StubAgentEnv.OBS_FOVEA, StubAgentEnv.OBS_PERIPHERAL],
      StubAgentEnv.OBS_FOVEA: cortex_f_config,
      StubAgentEnv.OBS_PERIPHERAL: cortex_p_config
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
    env_observation_space = self.env.observation_space

    # Build networks to preprocess the observation space
    default_config = self.get_default_config()  # TODO make this override
    with open(config_file) as json_file:
      delta_config = json.load(json_file)
      self._config = self.update_config(default_config, delta_config)

    obs_keys = self._config['obs_keys'] #[self.OBS_FOVEA, self.OBS_PERIPHERAL]
    self.modules = {}
    for obs_key in obs_keys:
      input_shape = self.create_input_shape(env_observation_space, obs_key)
      config = self._config[obs_key]
      cortex = PosteriorCortex(obs_key, input_shape, config)
      self.modules[obs_key] = cortex

    # input_config = {}
    # for stream in streams:
    #   env_obs = env_observation_space[stream]
    #   c_in = env_obs.shape[0]
    #   h = env_obs.shape[1]
    #   w = env_obs.shape[2]
    #   input_config[stream] = [-1, c_in, h, w]
    #   print('input config[',stream,':',input_config[stream])

    ############################################
    #channels = 3
    #self.retina = Retina(channels, config=None)
    #self.module.add_module('retina', self.retina)
    # c_in = obs_fovea.shape[0]
    # h = obs_fovea.shape[1]
    # w = obs_fovea.shape[2]
    # fovea_output_size = self.retina.get_output_shape(h, w)
    # c_out = 2 * channels  # because 2x 3 channels (+/-)
    # print('>>>>>>>>>>>>>>>>>>fovea fovea_output_size = ', fovea_output_size)
    # obs_shape_fovea = [6,34,34]#obs_fovea.shape
    # obs_space_fovea = spaces.Box(low=-math.inf, high=math.inf, shape=obs_shape_fovea, dtype=np.float32)
    #self.observation_space = obs_space_fovea
    ############################################

    # Build the new observation space dict from the cortically processed streams
    obs_dict = {}
    for obs_key in obs_keys:
      cortex = self.modules[obs_key]
      output_shape = cortex.get_output_shape()
      obs_shape = self.create_observation_shape(output_shape)   
      obs_space = self.create_observation_space(obs_shape)
      obs_dict[obs_key] = obs_space
    self.observation_space = spaces.Dict(obs_dict)

  @staticmethod
  def create_input_shape(env_observation_space, env_obs_key):
    env_obs = env_observation_space[env_obs_key]
    c_in = env_obs.shape[0]
    h = env_obs.shape[1]
    w = env_obs.shape[2]
    input_shape = [-1, c_in, h, w]
    return input_shape

  # @staticmethod
  # def create_input_config(self, env_observation_space, env_observation_keys):
  #   input_config = {}
  #   for env_observation_key in env_observation_keys:
  #     env_obs = env_observation_space[env_observation_key]
  #     c_in = env_obs.shape[0]
  #     h = env_obs.shape[1]
  #     w = env_obs.shape[2]
  #     input_config[env_observation_key] = [-1, c_in, h, w]
  #     print('input config[',env_observation_key,':',input_config[env_observation_key])
  #   return input_config

  def create_observation_space(self, observation_shape):
    observation_space = spaces.Box(low=-math.inf, high=math.inf, shape=observation_shape, dtype=np.float32)
    return observation_space

  def create_observation_shape(self, network_shape):
    b = network_shape[0]
    c = network_shape[1]
    h = network_shape[2]
    w = network_shape[3]
    observation_shape = [c,h,w]
    return observation_shape

  def reset(self):
    #print('>>>>>>>>>>> Stub reset')
    obs = self.env.reset()
    return self.forward(obs)

  def forward(self, observation):
    #print('-----------Obs old', observation)
    #obs_keys = [self.OBS_FOVEA, self.OBS_PERIPHERAL]
    obs_keys = self._config['obs_keys'] #[self.OBS_FOVEA, self.OBS_PERIPHERAL]

    obs_dict = {}
    for obs_key in obs_keys:
      cortex = self.modules[obs_key]
      input_tensor = self.obs_to_tensor(observation, obs_key)
      encoding_tensor, decoding, target = cortex.forward(input_tensor)
      self.tensor_to_obs(encoding_tensor, obs_dict, obs_key)
    return obs_dict

  #return output_f, output_p
  # obs_fovea = torch.tensor(observation['fovea'])
  # obs_fovea = torch.unsqueeze(obs_fovea, 0)  # insert batch dimension 0
  # print('!!!!!!!!!!!!!!!!! fovea shape:', obs_fovea.shape)
  # #dog_pos_fovea, dog_neg_fovea = self.retina(obs_fovea)
  # # dog_pos_fovea = dog_pos_fovea.to(device)
  # # dog_neg_fovea = dog_neg_fovea.to(device)
  # #print('dog shape:', dog_pos_fovea.shape)
  # #retina_fovea = torch.cat([dog_pos_fovea, dog_neg_fovea], 1)
  # retina_fovea = torch.squeeze(retina_fovea).detach().numpy()  # remove batch dim
  # print('>>>>>>> final obs shape:', retina_fovea.shape)
  #print('-----------Obs new', tx_observation)
  #return retina_fovea

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
    #print('>>>>>>>>>>> Stub step')
    [obs, self.reward, is_end_state, additional] = self.env.step(action)
    tx_obs = self.forward(obs)
    emit = [tx_obs, self.reward, is_end_state, additional]
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
