import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import pygame as pygame
from .pygame_env import PyGameEnv

class CountingEnv(PyGameEnv):

  def __init__(self):
    w = 300
    h = 200
    a = 2
    super().__init__(a, w, h)

  def reset(self):
    self.state = [0]

  def step(self, action):
    #return ob, reward, is_complete, info
    old_state = self.state
    if action == 0:
      delta = -1
    else:
      delta = 1
    state = old_state[0] +1 +delta

    reward = 0.0
    if (state == 5) == 0:
      reward = 1.0

    done = False
    min_state = 0
    max_state = 9
    if state < min_state:
      state = min_state
    if state >= max_state:
      state = max_state
      done = True

    new_state = [state]

    #ob = self._get_obs()
    self.state = new_state
    self.reward = reward
    self.done = done
    self.add = {
      'old_state': old_state,
      'new_state': new_state,
      'reward': self.reward, 
      'action': action,
      'done': done }
    print('additional:', str(self.add))
    return [self.state, self.reward, self.done, self.add]
   
  def _get_image(self):    
    image = np.zeros((100,100,3))
    for i in range(0, 30):
      for j in range(0, 30):
        image[i][j][0] = 100.0
    return image
