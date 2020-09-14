import gym
from gym import error, spaces, utils
from gym.utils import seeding

from abc import ABC, abstractmethod
import numpy as np
import os
import sys
import csv
import random
import pygame as pygame
from pygame.locals import *

from .pygame_env import PyGameEnv
from .active_vision_env import ActiveVisionEnv

#class FiniteStateEnv(PyGameEnv):
class FiniteStateEnv(ActiveVisionEnv):

  def __init__(self, num_actions, screen_width, screen_height, frame_rate=30):

    # Create state machine
    self.states = {}
    self.start_state = None
    self.state_key = None
    self._create_states()

    super().__init__(num_actions, screen_width, screen_height, frame_rate)

  @abstractmethod
  def _create_states(self):
    """Override this method to define the state-graph"""
    pass

  @abstractmethod
  def _update_state_key(self, old_state_key, action, elapsed_time):
    """
    Return the new state key, or the old state key if unchanged.
    :rtype: state_key
    """
    pass

  @abstractmethod
  def _update_reward(self, old_state_key, action, elapsed_time, new_state_key):
    """
    Return the reward for the specified old-state/new-state given action,time.
    :rtype: real
    """
    pass

  def reset(self):
    assert(self.start_state is not None)
    self.state_key = None
    self.state_time = None
    self.set_state(self.start_state)
    return super().reset()

  def add_state(self, state_key, start_state=False, end_state=False, next_states=[], duration=None, meta=None):
    state = {
      'key':state_key,
      'start':start_state,
      'end':end_state,
      'duration':duration,
      'next_state_keys':next_states,
      'meta':meta
    }    
    self.states[state_key] = state
    if start_state is True:
      self.start_state = state_key
      #print('Start state = ', state_key)
    #print('Adding state:', str(state))

  def set_state(self, state_key):
    old_state_key = self.state_key
    time = self.get_time()  # Record the time of entering the state
    self.state_key = state_key  # The key, not the object
    self.state_time = time  # When we entered the state
    self.on_state_changed(old_state_key, state_key)

  def on_state_changed(self, old_state_key, new_state_key):
    print('State -> ', state_key, '@t=', self.state_time)

  def get_state_key(self):
    return self.state_key

  def get_state_elapsed_time(self):
    time = self.get_time()
    elapsed_time = time - self.state_time
    return elapsed_time

  def is_end_state(self, state_key):
    state = self.states[state_key]
    return state['end']

  def get_next_state_keys(self, state_key):
    state = self.states[state_key]
    next_states = state['next_state_keys']
    return next_states

  def _do_step(self, action, time):
    old_state_key = self.get_state_key()
    elapsed_time = self.get_state_elapsed_time()
    #print('old state=', old_state_key, 'time=',elapsed_time)
    new_state_key = self._update_state_key(old_state_key, action, elapsed_time)
    #print('new state=', new_state_key)
    reward = self._update_reward(old_state_key, action, elapsed_time, new_state_key)

    if new_state_key != old_state_key:
      self.set_state(new_state_key)

    observation = self.get_observation()
    is_end_state = self.is_end_state(new_state_key)

    #ob = self._get_obs()
    self.reward = reward
    additional = {
      'old_state': old_state_key,
      'new_state': new_state_key,
      'reward': self.reward, 
      'action': action,
      'done': is_end_state }
    print('additional:', str(additional))
    return [observation, self.reward, is_end_state, additional]
