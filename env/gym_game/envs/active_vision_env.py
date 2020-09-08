import gym
from gym import error, spaces, utils
from gym.utils import seeding

from abc import ABC, abstractmethod
import numpy as np
import pygame as pygame

from .pygame_env import PyGameEnv

class ActiveVisionEnv(PyGameEnv):

  metadata = {'render.modes': ['human','rgb_array']}

  HUMAN = 'human'
  ARRAY = 'rgb_array'

  def __init__(self, num_actions, screen_width, screen_height, frame_rate=30):
    super().__init__(num_actions, screen_width, screen_height, frame_rate)
    self.gaze = []

  def reset(self):
    """Reset gaze coordinates"""
    super().reset()

  def _create_action_space(self, num_actions):
    #total_actions = num_actions + 2  # Gaze control
    self.action_space = spaces.Discrete(num_actions)

  def _create_observation_space(self, screen_width, screen_height, channels=3, dtype=np.uint8):    
    full = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, channels), dtype=dtype)
    fovea = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, channels), dtype=dtype)
    peripheral = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, channels), dtype=dtype)
    self.observation_space = spaces.Dict({
      'full':full, 
      'fovea':fovea, 
      'peripheral':peripheral})

  def get_observation(self):
    print('Obs DICT')
    img = self.render(mode='rgb_array')
    observation = {
      'full':img, 
      'fovea':img, 
      'peripheral':img}
    return observation
