import math
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from abc import ABC, abstractmethod
import numpy as np
import pygame as pygame

from gym_game.stubs.retina import Retina
# from utils.medial_temporal_lobe import MedialTemporalLobe
# from utils.positional_encoding import PositionalEncoder
# from utils.prefrontal_cortex import PrefrontalCortex
# from utils.superior_colliculus import SuperiorColliculus

from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()

"""
  Wraps a task-specific environment and implements brain modules that are not trained by Reinforcement Learning.
"""

class StubAgentEnv(gym.Env):

  # default config
  default_config = {
    "retina": {
      'f_size': 7,
      'f_sigma': 2.0,
      'f_k': 1.6  # approximates Laplacian of Gaussian
    },
    "positional_encoding": {},
    "vc_fovea": {},
    "vc_periphery": {},
    "mtl": {},
    "sc": {},
    "pfc": {}
  }

  def __init__(self, env_type, env_config_file):
    self.env = gym.make(env_type, config_file=env_config_file)
    #self.env = env_type(env_config_file)
    self.action_space = self.env.action_space
    env_observation_space = self.env.observation_space

    # Build a module to hold sub-components
    self.module = nn.Module()

    # Build networks to preprocess the observation space
    obs_fovea = env_observation_space['fovea']
    channels = 3
    self.retina = Retina(channels, config=None)
    self.module.add_module('retina', self.retina)

    c_in = obs_fovea.shape[0]
    h = obs_fovea.shape[1]
    w = obs_fovea.shape[2]
    fovea_output_size = self.retina.get_output_shape(h, w)
    c_out = 2 * channels  # because 2x 3 channels (+/-)
    print('>>>>>>>>>>>>>>>>>>fovea fovea_output_size = ', fovea_output_size)

    obs_shape_fovea = [6,34,34]#obs_fovea.shape
    obs_space_fovea = spaces.Box(low=-math.inf, high=math.inf, shape=obs_shape_fovea, dtype=np.float32)

    self.observation_space = obs_space_fovea

  def reset(self):
    print('>>>>>>>>>>> Stub reset')
    obs = self.env.reset()
    return self.forward(obs)

  def forward(self, observation):
    #print('-----------Obs old', observation)
    obs_fovea = torch.tensor(observation['fovea'])
    obs_fovea = torch.unsqueeze(obs_fovea, 0)  # insert batch dimension 0
    print('!!!!!!!!!!!!!!!!! fovea shape:', obs_fovea.shape)
    dog_pos_fovea, dog_neg_fovea = self.retina(obs_fovea)
    # dog_pos_fovea = dog_pos_fovea.to(device)
    # dog_neg_fovea = dog_neg_fovea.to(device)
    print('dog shape:', dog_pos_fovea.shape)
    retina_fovea = torch.cat([dog_pos_fovea, dog_neg_fovea], 1)
    retina_fovea = torch.squeeze(retina_fovea).detach().numpy()  # remove batch dim
    print('>>>>>>> final obs shape:', retina_fovea.shape)

    #print('-----------Obs new', tx_observation)
    return retina_fovea

  def get_config(self):
    """ return a dictionary of params """
    return self.env.get_config()

  def step(self, action):
    print('>>>>>>>>>>> Stub step')
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
