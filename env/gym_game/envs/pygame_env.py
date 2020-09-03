import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import pygame as pygame

class PyGameEnv(gym.Env):

  metadata = {'render.modes': ['human','rgb_array']}

  HUMAN = 'human'
  ARRAY = 'rgb_array'

  def __init__(self, num_actions, screen_width, screen_height):
    # here spaces.Discrete(2) means that action can either be L or R choice
    self._create_action_space(num_actions)
    self._create_observation_space(screen_width, screen_height)
    self.seed()  # Ensure repeatability of game
    self.reset()
    self.screen = None

  def reset(self):
    """Reset the game to a valid initial state."""
    pass

  def step(self, action):
    """Update the game-state given the provided action."""
    pass

  def _create_action_space(self, num_actions):
    self.action_space = spaces.Discrete(num_actions)

  def _create_observation_space(self, screen_width, screen_height, channels=3, dtype=np.uint8):    
    self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, channels), dtype=dtype)

  def get_observation_shape(self):
    return self.observation_space.shape

  def get_random(self):
    """Return the PRNG for this game"""
    return self.np_random

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    self.seed = seed
    return seed

  def get_keys_pressed(self):
    pressed = pygame.key.get_pressed()
    pygame.event.pump()
    return pressed

  def get_time(self):
    time = pygame.time.get_ticks()
    return time

  def get_events(self):
    events = pygame.event.get()
    return events

  def render_image(self):
    image = np.zeros((100,50,3))
    for i in range(0, 30):
      for j in range(0, 30):
        image[i][j][0] = 100.0
    return image

  def _get_window_name(self):
    return "PyGame Env"

  def _get_background_colour(self):
    background_colour = (0,0,0)
    return background_colour

  def render(self, mode='human', close=False):
    """Renders the game to a window, or as an image. Override render_image() to define what's shown. Close is a hint that the window shoud b"""
    img = self.render_image()
    if mode == 'rgb_array':
      return img, True
    elif mode == 'human':
      try:
        import pygame
        from pygame import gfxdraw
      except ImportError as e:
        raise error.DependencyNotInstalled(
          "{}. (HINT: install pygame using `pip install pygame`".format(e))
      if close:
        pygame.quit()
      else:
        observation_shape = self.get_observation_shape()
        self.window_height = observation_shape[0]  # height
        self.window_width = observation_shape[1]  # width
        if self.screen is None:
          pygame.init()
          clock = pygame.time.Clock()
          self.screen = pygame.display.set_mode(
              (round(self.window_width), round(self.window_height)))
          pygame.display.set_caption(self._get_window_name())
          background_colour = self._get_background_colour()
          self.screen.fill(background_colour)
          surface = pygame.surfarray.make_surface(img)
          self.screen.blit(surface, (0, 0))  # Draw game state
          pygame.display.update()
      return img, True
    else:
      raise error.UnsupportedMode("Unsupported render mode: " + mode)      
