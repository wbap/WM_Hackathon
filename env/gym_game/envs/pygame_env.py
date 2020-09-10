import gym
from gym import error, spaces, utils
from gym.utils import seeding

from abc import ABC, abstractmethod
import numpy as np
import pygame as pygame


class PyGameEnv(gym.Env, ABC):

  metadata = {'render.modes': ['human','rgb_array']}

  HUMAN = 'human'
  ARRAY = 'rgb_array'

  def __init__(self, num_actions, screen_width, screen_height, frame_rate=30):
    # here spaces.Discrete(2) means that action can either be L or R choice
    self._create_action_space(num_actions)
    self._create_observation_space(screen_width, screen_height)
    self.seed()  # Ensure repeatability of game
    self.reset()
    self.display_screen = None
    pygame.init()
    self.screen_shape = [screen_height, screen_width, 3]
    self.screen = pygame.Surface((screen_width,screen_height))  # draw on here
    self.frame_rate = frame_rate

  def reset(self):
    """Reset the game to a valid initial state."""
    self.clock = pygame.time.Clock()

  def get_time(self):
    """Returns game time in milliseconds"""
    return self.clock.get_ticks()

  @abstractmethod
  def _do_step(self, action, time):
    pass

  def step(self, action):
    """Update the game-state given the provided action."""
    time = self.clock.tick()
    return self._do_step(action, time)

  def _create_action_space(self, num_actions):
    self.action_space = spaces.Discrete(num_actions)

  def _create_observation_space(self, screen_width, screen_height, channels=3, dtype=np.uint8):    
    self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, channels), dtype=dtype)

  def get_screen_shape(self):
    return self.screen_shape

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

  def _get_caption(self):
    return "PyGame Env"

  def _get_background_colour(self):
    background_colour = (0,0,0)
    return background_colour

  def get_observation(self):
    img = self.render(mode='rgb_array')
    # TODO: render returns a tuple, is img meant to be a tuple - i.e. is this a bug?
    return img

  def render(self, mode='human', close=False):
    """
    Renders the game to a window, or as an image.
    Override render_image() to define what's shown.
    Close is a hint that the window should b
    """
    try:
      import pygame
      from pygame import gfxdraw
    except ImportError as e:
      raise error.DependencyNotInstalled(
        "{}. (HINT: install pygame using `pip install pygame`".format(e))
    if close:
      pygame.quit()

    background_colour = self._get_background_colour()
    self.screen.fill(background_colour)

    #img = self.render_image()
    #surface = pygame.surfarray.make_surface(img)
    #self.screen.blit(surface, (0, 0))  # Draw game state
    self.render_screen(self.screen)

    if mode == 'rgb_array':
      pass
    elif mode == 'human':
      #print('HUMAN')
      if self.display_screen is None:
        screen_shape = self.get_screen_shape()
        self.window_height = screen_shape[0]  # height
        self.window_width = screen_shape[1]  # width
        self.display_screen = pygame.display.set_mode(
            (round(self.window_width), round(self.window_height)))
        pygame.display.set_caption(self._get_caption())

      # Update the window
      #surface = pygame.surfarray.make_surface(img)
      #self.screen.blit(surface, (0, 0))  # Draw game state
      self.display_screen.blit(self.screen, (0, 0))  # Draw game state
      pygame.display.update()
    else:
      raise error.UnsupportedMode("Unsupported render mode: " + mode)      

    # Convert to numpy array
    rgb_array = pygame.surfarray.array3d(self.screen)
    return rgb_array, True

  def render_screen(self, screen):
    pass

  # def render_image(self):
  #   image = np.zeros((100,50,3))
  #   for i in range(0, 30):
  #     for j in range(0, 30):
  #       image[i][j][0] = 100.0
  #   return image

