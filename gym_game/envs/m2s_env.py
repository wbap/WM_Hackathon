import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import os
import sys
import csv
import random
import pygame as pygame
from pygame.locals import *

from .pygame_env import PyGameEnv
from .finite_state_env import FiniteStateEnv
from .dm2s_env import Dm2sEnv


class M2sEnv(Dm2sEnv):

  def __init__(self, config_file=None):
    super().__init__(config_file)

  def _create_states(self):
    def get_interval(state):
      interval = self.gParams['states'][state]['interval']
      return interval

    #self.add_state(self.STATE_TUTOR_STIM, next_states=[self.STATE_TUTOR_SHOW], duration=show_stim_interval, start_state=True)
    self.add_state(self.STATE_TUTOR_SHOW, next_states=[self.STATE_TUTOR_FEEDBACK], duration=get_interval(self.STATE_TUTOR_SHOW), start_state=True)
    self.add_state(self.STATE_TUTOR_FEEDBACK, next_states=[self.STATE_INTER], duration=get_interval(self.STATE_TUTOR_FEEDBACK))

    #self.add_state(self.STATE_INTER, next_states=[self.STATE_PLAY_STIM], duration=inter_interval)
    self.add_state(self.STATE_INTER, next_states=[self.STATE_PLAY_SHOW], duration=get_interval(self.STATE_INTER))

    #self.add_state(self.STATE_PLAY_STIM, next_states=[self.STATE_PLAY_SHOW], duration=show_stim_interval)
    self.add_state(self.STATE_PLAY_SHOW, next_states=[self.STATE_PLAY_FEEDBACK], duration=get_interval(self.STATE_PLAY_SHOW))
    self.add_state(self.STATE_PLAY_FEEDBACK, next_states=[self.STATE_END], duration=get_interval(self.STATE_PLAY_FEEDBACK))

    self.add_state(self.STATE_END, end_state=True)

  def _get_caption(self):
    return 'Match-to-Sample'

  def on_state_changed(self, old_state_key, new_state_key):
    #print('State -> ', new_state_key, '@t=', self.state_time)
    #print('------------------------------------------------------------------ State -> ', new_state_key, '@t=', self.state_time)
    if new_state_key == self.STATE_TUTOR_SHOW or new_state_key == self.STATE_PLAY_SHOW:
      self.get_random_samples()

  def _update_state_key(self, old_state_key, action, elapsed_time):
    #print('Env Step')

    # Don't transition from end states
    is_end_state = self.is_end_state(old_state_key)
    if is_end_state:
      return old_state_key  # terminal state

    # transition on response, when appropriate
    new_state_key = old_state_key  # default
    next_state_keys = self.get_next_state_keys(old_state_key)
    if old_state_key == self.STATE_PLAY_SHOW:
      if action in [self.ACTION_LEFT, self.ACTION_RIGHT]:
        if self.result is None:
          if action == self.position:
            self.result = self.RESULT_CORRECT
          else:
            self.result = self.RESULT_WRONG
          print("Response: "+str(action)+", Correct response: "+str(self.position))
        new_state_key = next_state_keys[0]
        return new_state_key
      # else: wait for timer

    # All other states - check timer
    state = self.states[old_state_key]
    duration = state['duration']
    if duration is not None:
      if elapsed_time > duration:
        new_state_key = next_state_keys[0]

        if old_state_key == self.STATE_PLAY_SHOW:
          print("Response: None, Correct response: "+str(self.position))

        # Repeat certain sections again and again        
        self.tutor_repeats = int(self.gParams["observationRepeat"])
        self.play_repeats = int(self.gParams["mainTaskRepeat"])
        if old_state_key == self.STATE_TUTOR_FEEDBACK:
          self.tutor_counts += 1
          if self.tutor_counts < self.tutor_repeats:
            #new_state_key = self.STATE_TUTOR_STIM
            new_state_key = self.STATE_TUTOR_SHOW
        elif old_state_key == self.STATE_PLAY_FEEDBACK:
          self.play_counts += 1
          if self.play_counts < self.play_repeats:
            #new_state_key = self.STATE_PLAY_STIM
            new_state_key = self.STATE_PLAY_SHOW
        return new_state_key
    return old_state_key

  def get_screen_options(self, state, elapsed_time):
    screen_options = super().get_screen_options(state, elapsed_time)
    if state == self.STATE_TUTOR_SHOW or state == self.STATE_PLAY_SHOW:
      screen_options['sample'] = True 
    return screen_options
