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

class Dm2sEnv(FiniteStateEnv):

  # Define states of the game
  STATE_TUTOR_STIM = 'tutor-stim'
  STATE_TUTOR_HIDE = 'tutor-hide'
  STATE_TUTOR_SHOW = 'tutor-show'
  STATE_TUTOR_RESPONSE = 'tutor-response'
  STATE_INTER = 4
  STATE_PLAY_STIM = 4
  STATE_PLAY_HIDE = 4
  STATE_PLAY_SHOW = 4
  STATE_PLAY_RESPONSE = 'play-response'
  STATE_END = 'end'

  # Define actions
  ACTION_NONE = 0
  ACTION_LEFT = 1
  ACTION_RIGHT = 2
  NUM_ACTIONS = 3

  # define colors
  BLACK = (0,0,0)
  WHITE = (255, 255, 255)
  YELLOW = (255,255,0)
  BLUE = (0,0,255)

  # define global variables
  gParams = {}  # parameter dictionary
  gVideoWidth = 800
  gVideoHeight = 800
  gColors =  ["LB"]
  gShapes =  ["Barred_Ring","Triangle","Crescent","Cross","Circle","Heart","Pentagon","Ring","Square"]
  gButton1 = None
  gButton1F = None
  gButton2 = None
  gButton2F = None
  gCorrectFB = Rect(0, gVideoHeight - 80, gVideoWidth, 80)
  gObservBar = Rect(0, 0, gVideoWidth, 80)

  def __init__(self, config_file=None):
    # obtain parameters from a file
    print('Config file:', config_file)
    with open(config_file) as f:
      for line in f:
        buf = line.strip().split(",")
        self.gParams[buf[0]] = buf[1]
    print('gParams:', self.gParams)

    # load button images
    self.gButton1 = pygame.image.load('png/Button_Green.png')
    self.gButton1F = pygame.image.load('png/Button_LG.png')
    self.gButton2 = pygame.image.load('png/Button_Red.png')
    self.gButton2F = pygame.image.load('png/Button_Pink.png')

    w = self.gVideoWidth
    h = self.gVideoHeight

    # Create base class stuff
    num_actions = 3  # No action, Left and Right
    super().__init__(self.NUM_ACTIONS, w, h)

  def _create_states(self):
    print('create states')
    self.add_state(self.STATE_TUTOR_STIM, next_states=[self.STATE_TUTOR_HIDE], duration=1, start_state=True)
    self.add_state(self.STATE_TUTOR_HIDE, next_states=[self.STATE_TUTOR_SHOW], duration=1)
    self.add_state(self.STATE_TUTOR_SHOW, next_states=[self.STATE_TUTOR_RESPONSE], duration=1)
    self.add_state(self.STATE_TUTOR_RESPONSE, next_states=[self.STATE_END], duration=1)
    self.add_state(self.STATE_END, end_state=True)

  def _get_caption(self):
    return 'Delayed Match-to-Sample'

  def reset(self):
    super().reset()

  def _update_state_key(self, old_state_key, action, elapsed_time):
    # Don't transition from end states
    is_end_state = self.is_end_state(old_state_key)
    if is_end_state:
      return old_state_key  # terminal state

    # transition on response, when appropriate
    new_state_key = old_state_key  # default
    next_state_keys = self.get_next_state_keys(old_state_key)
    if old_state_key == self.STATE_TUTOR_RESPONSE:
      if action != self.ACTION_NONE:
        new_state_key = next_state_keys[0]
        return new_state_key

    # check timer
    state = self.states[old_state_key]
    duration = state['duration']
    print('duration', duration, 'elapsed', elapsed_time)
    if duration is not None:
      if elapsed_time > duration:
          new_state_key = next_state_keys[0]
          return new_state_key
    return old_state_key

  def _update_reward(self, old_state_key, action, elapsed_time, new_state_key):
    reward = 0.0
    if old_state_key == self.STATE_PLAY_RESPONSE:
      if action == self.ACTION_LEFT:
        reward = 1.0
      elif action == self.ACTION_RIGHT:
        reward = 1.0
    return reward

  def render_screen(self, screen):
    screen.fill(self.WHITE)
