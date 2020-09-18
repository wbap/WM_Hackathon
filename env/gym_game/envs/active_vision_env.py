import logging

import numpy as np
from gym import spaces

from .pygame_env import PyGameEnv


class ActiveVisionEnv(PyGameEnv):
  metadata = {'render.modes': ['human', 'rgb_array']}

  HUMAN = 'human'
  ARRAY = 'rgb_array'

  GAZE_STEP_SIZE = 100  # number of pixels, step size of gaze movement (commanded as an action direction)
  FOVEA_FRACTION = 0.1  # fraction of diameter, at centre of image

  # i = 0

  def __init__(self, num_actions, screen_width, screen_height, frame_rate=30):
    self.fov_fraction = self.FOVEA_FRACTION
    self.step_size = self.GAZE_STEP_SIZE
    self.fov_size = np.array([int(self.FOVEA_FRACTION * screen_width), int(self.FOVEA_FRACTION * screen_height)])
    self.gaze = np.array([screen_width // 2, screen_height // 2])  # gaze position - (x, y) tuple
    self._action_2_xy = {  # map actions (integers) to x,y gaze delta
      num_actions: np.array([-1, 0]),      # 3. left
      num_actions + 1: np.array([1, 0]),   # 4. right
      num_actions + 2: np.array([0, -1]),  # 5. up
      num_actions + 3: np.array([0, 1])    # 6. down
    }
    self._actions_start = num_actions
    self._actions_end = num_actions + len(self._action_2_xy)

    self._img_fov = None
    self._img_periph = None
    self._last_observed_gaze = self.gaze

    self._x_min = self.fov_size[0]
    self._x_max = screen_width - self.fov_size[0]
    self._y_min = self.fov_size[1]
    self._y_max = screen_height - self.fov_size[1]

    super().__init__(num_actions, screen_width, screen_height, frame_rate)

  def reset(self):
    """Reset gaze coordinates"""
    super().reset()

  def _do_step(self, action, time):
    # update the position of the fovea (fov_pos), given the action taken
    logging.debug("Received action: ", action)
    # if within action scope, modify gaze
    if self._actions_start <= action < self._actions_end:
      self.gaze = self.gaze + self._action_2_xy[action] * self.step_size

      self.gaze[0] = np.clip(self.gaze[0], self._x_min, self._x_max)   # ensure x coord is in bounds
      self.gaze[1] = np.clip(self.gaze[1], self._y_min, self._y_max)   # ensure y coord is in bounds

      print("New gaze: ", self.gaze)

  def _create_action_space(self, num_actions):
    total_actions = num_actions + self._actions_end  # Gaze control
    self.action_space = spaces.Discrete(total_actions)

  def _create_observation_space(self, screen_width, screen_height, channels=3, dtype=np.uint8):
    full = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, channels), dtype=dtype)
    fovea = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, channels), dtype=dtype)
    peripheral = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, channels), dtype=dtype)
    self.observation_space = spaces.Dict({
      'full': full,
      'fovea': fovea,
      'peripheral': peripheral})

  def get_observation(self):

    img = self.render(mode='rgb_array')
    img = np.transpose(img, [1, 0, 2])  # observed img is horizontally reflected, and rotated 90 deg ...

    # if gaze position changed, then update foveal and peripheral images, unless they are undefined
    imgs_undef = self._img_fov is None or self._img_periph is None
    gaze_changed = (self._last_observed_gaze != self.gaze).any()
    if imgs_undef or gaze_changed:
      # crop to fovea (centre region)
      h, w, ch = img.shape[0], img.shape[1], img.shape[2]

      pixels_h = int(h * self.fov_fraction)
      pixels_w = int(w * self.fov_fraction)

      self._img_fov = img[self.gaze[1]:self.gaze[1] + pixels_h, self.gaze[0]:self.gaze[0] + pixels_w, :]

      # zero foveal region, and return as peripheral (other options are to blur or cut it up differently)
      self._img_periph = img.copy()
      self._img_periph[self.gaze[1]:self.gaze[1] + pixels_h, self.gaze[0]:self.gaze[0] + pixels_w, :] = 0.

      self._last_observed_gaze = self.gaze.copy()

      # debugging
      # plt.imsave(str(self.i)+'_fov.png', self.img_fov)
      # plt.imsave(str(self.i)+'_periph.png', self.img_periph)
      # self.i += 1

    observation = {
      'full': img,
      'fovea': self._img_fov,
      'peripheral': self._img_periph}

    return observation
