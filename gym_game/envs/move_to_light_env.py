import json
import pygame as pygame
import numpy as np
import logging

from .finite_state_env import FiniteStateEnv

near_target_radius = 0


class MoveToLightEnv(FiniteStateEnv):
  """
  A simple game. A colour light is shown, and the gaze must move to the light.
  """

  # Define states of the game
  STATE_IDLE = 'idle'
  STATE_SHOW_TARGET = 'show_target'  # a colour target appears
  STATE_ON_TARGET = 'on_target'      # moved close enough to target
  STATE_END = 'end'                  # end of game

  # Define actions
  ACTION_NONE = 0
  NUM_ACTIONS = 1  # action=0 is 'no action'. Additional actions will be added by base classes if necessary.

  RESULT_WRONG = 0
  RESULT_CORRECT = 1

  # define colors
  BLACK = (0, 0, 0)
  WHITE = (255, 255, 255)
  YELLOW = (255, 255, 0)
  BLUE = (0, 0, 255)
  GRAY = (128, 128, 128)
  RED = (255, 0, 0)

  # define global variables
  config = {}  # parameter dictionary

  def __init__(self, config_file=None):
    # obtain parameters from a file
    print('Env config file:', config_file)

    with open(config_file) as json_file:
      self.config = json.load(json_file)
    print('MoveToLightEnv config:', self.config)

    self.gVideoWidth = self.config["videoWidth"]
    self.gVideoHeight = self.config["videoHeight"]
    self.play_repeats = self.config["mainTaskRepeat"]
    self.target_radius = self.config["targetRadius"]

    w = self.gVideoWidth
    h = self.gVideoHeight

    self.target_centre = None    # x, y
    self.result = None
    self.play_counts = 0

    self._show_fovea = True if int(self.config["show_fovea_on_screen"]) else False

    # Create base class stuff
    frame_rate = int(self.config["frameRate"])
    super().__init__(self.NUM_ACTIONS, w, h, frame_rate)

  def get_config(self):
    return self.config

  def _create_states(self):
    # note, the first 'next_state' is the default state to transition to, when interval has elapsed

    idle_duration = 0
    light_duration = 10000
    on_target_duration = 1000

    self.add_state(self.STATE_IDLE, next_states=[self.STATE_SHOW_TARGET], duration=idle_duration, start_state=True)
    self.add_state(self.STATE_SHOW_TARGET, next_states=[self.STATE_IDLE, self.STATE_ON_TARGET], duration=light_duration)
    self.add_state(self.STATE_ON_TARGET, next_states=[self.STATE_SHOW_TARGET], duration=on_target_duration)
    self.add_state(self.STATE_END, end_state=True)

  def _get_caption(self):
    return 'Move To Light'

  def reset(self):
    self.target_centre = np.array([self.gVideoWidth // 2, self.gVideoHeight // 2])
    self.result = None
    self.play_counts = 0
    return super().reset()

  def on_state_changed(self, old_state_key, new_state_key):
    print('State -> ', new_state_key, '@t=', self.state_time)

    if new_state_key == self.STATE_SHOW_TARGET:
      self.target_centre = self.get_random_sample()

  def _update_state_key(self, old_state_key, action, elapsed_time):
    """
      Return the new state key, or the old state key if unchanged.
      Actions do not directly cause state changes in this game
      State changes occur by position of fovea, or via timer
    """

    # Don't transition from end states
    is_end_state = self.is_end_state(old_state_key)
    if is_end_state:
      return old_state_key  # terminal state

    next_state_keys = self.get_next_state_keys(old_state_key)

    # Check if on target
    if old_state_key != self.STATE_ON_TARGET and self.is_on_target():
      new_state_key = self.STATE_ON_TARGET
      return new_state_key

    # Otherwise, check timer
    state = self.states[old_state_key]
    duration = state['duration']
    if duration is not None:
      if elapsed_time > duration:
        new_state_key = next_state_keys[0]

        # continue showing new targets (until max reached)
        if old_state_key == self.STATE_SHOW_TARGET:
          self.play_counts += 1
          if self.play_counts > self.play_repeats:
            new_state_key = self.STATE_END
        return new_state_key

    return old_state_key

  def is_on_target(self):
    touching_distance = self.target_radius + self.fov_size_half[0]  # assumes fovea is equal size in both x and y
    distance = np.sqrt(np.sum(np.square(self.gaze_centre - self.target_centre)))
    gap = distance - touching_distance

    print(" ** gap calc ** touching_dist = {}, distance = {}, gap = {} ".format(touching_distance, distance, gap))

    if gap <= near_target_radius:
      return True
    else:
      return False

  def _update_reward(self, old_state_key, action, elapsed_time, new_state_key):
    """ At the end of game (timed-out or reached target), give the +ve or -ve reward """
    reward = 0.0
    if old_state_key == self.STATE_SHOW_TARGET and new_state_key == self.STATE_ON_TARGET:   # reached target
      reward = 1.0
      print(" -------- Reward = 1")
    elif old_state_key == self.STATE_SHOW_TARGET and new_state_key == self.STATE_IDLE:      # timed out
      reward = -1.0
      print(" ++++++++ Reward = -1")
    return reward

  def get_random_sample(self):

    mode_grid = True

    if mode_grid:
      grid_cell = self.np_random.randint(0, self.grid_utils.num_cells()-1)

      # convert to action space
      action = grid_cell + self._actions_start

      # convert to xy
      xy = self.grid_utils.action_2_xy(action)
      x = xy[0]
      y = xy[1]

    else:   # continuous
      min_xy = self.target_radius
      max_x = self.screen_shape[0] - self.target_radius
      max_y = self.screen_shape[1] - self.target_radius

      y = self.np_random.randint(min_xy, max_x)
      x = self.np_random.randint(min_xy, max_y)

    return np.array([x, y])

  def render_screen(self, screen):
    state = self.get_state_key()
    if state == self.STATE_END:
      screen.fill(self.BLACK)
      return

    elapsed_time = self.get_state_elapsed_time()

    # fill screen
    screen.fill(self.WHITE)
    screen_options = self.get_screen_options(state, elapsed_time)
    self.draw_screen(screen, screen_options)

  def get_screen_options(self, state, elapsed_time):
    screen_options = self.get_default_screen_options()
    if state == self.STATE_SHOW_TARGET:
      screen_options['target'] = True
    return screen_options

  def get_default_screen_options(self):
    screen_options = {
      'target': False
    }
    return screen_options

  def draw_screen(self, screen, screen_options):
    # fill screen
    screen.fill(self.WHITE)

    if screen_options['target']:
      target_rect = pygame.Rect(self.target_centre[0]-self.target_radius, self.target_centre[1]-self.target_radius,
                                2*self.target_radius, 2*self.target_radius)
      pygame.draw.rect(screen, self.RED, target_rect)

    super().draw_screen(screen, [])
