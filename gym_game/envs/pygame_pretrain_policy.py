import gym
from gym import error, spaces, utils
from gym.utils import seeding

from abc import ABC, abstractmethod
import numpy as np
import pygame as pygame

class PyGamePretrainPolicy():
  """Generates actions (policy) for a PyGame environment."""

  def __init__(self, action_space):
    if not hasattr(action_space, 'n'):
        raise Exception('Keyboard agent only supports discrete action spaces')
    self._action_space = action_space

  def reset(self):
    """Hook after episode or rollout change."""
    pass

  def get_action(self, observation):
    """Return an action from the action_space, given the observation."""
    num_actions = self._action_space.n
    action = np.random.randint(num_actions)
    return action
