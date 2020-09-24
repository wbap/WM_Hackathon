import gym
from gym import error, spaces, utils
from gym.utils import seeding

from abc import ABC, abstractmethod
import numpy as np
import pygame as pygame

"""
  Wraps a task-specific environment and implements brain modules that are not trained by Reinforcement Learning.
"""

class StubAgentEnv(gym.Env):

  def __init__(self, env_type, env_config_file):
    self.env = gym.make(env_type, config_file=env_config_file)
    #self.env = env_type(env_config_file)
    self.action_space = self.env.action_space
    env_observation_space = self.env.observation_space
    self.observation_space = env_observation_space['fovea']

  def reset(self):
    print('>>>>>>>>>>> Stub reset')
    obs = self.env.reset()
    return self.forward(obs)

  def forward(self, observation):
    #print('-----------Obs old', observation)
    tx_observation = observation['fovea']
    #print('-----------Obs new', tx_observation)
    return tx_observation

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
