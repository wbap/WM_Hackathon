import logging

import numpy as np
from gym import spaces

from .pygame_env import PyGameEnv


class ActiveVisionEnv(PyGameEnv):
  metadata = {'render.modes': ['human', 'rgb_array']}

  HUMAN = 'human'
  ARRAY = 'rgb_array'

  GAZE_STEP_SIZE = 20  # number of pixels, step size of gaze movement (commanded as an action direction)
  FOVEA_FRACTION = 0.1  # fraction of diameter, at centre of image

  # i = 0

  def __init__(self, num_actions, screen_width, screen_height, frame_rate=30):
    self.fov_fraction = self.FOVEA_FRACTION
    self.step_size = self.GAZE_STEP_SIZE
    self.gaze = np.array([screen_width // 2, screen_height // 2])  # gaze position - (x, y) tuple
    self.action_2_xy = {  # map actions (integers) to x,y gaze delta
      num_actions: np.array([-1, 0]),  # 3. left
      num_actions + 1: np.array([1, 0]),  # 4. right
      num_actions + 2: np.array([0, -1]),  # 5. up
      num_actions + 3: np.array([0, 1])  # 6. down
    }
    self.actions_start = num_actions
    self.actions_end = num_actions + len(self.action_2_xy)

    self.img_fov = None
    self.img_periph = None
    self._last_observed_gaze = self.gaze

    super().__init__(num_actions, screen_width, screen_height, frame_rate)

  def reset(self):
    """Reset gaze coordinates"""
    super().reset()

  def _do_step(self, action, time):
    # update the position of the fovea (fov_pos), given the action taken
    logging.debug("Received action: ", action)
    # if within action scope, modify gaze
    if self.actions_start <= action < self.actions_end:
      self.gaze = self.gaze + self.action_2_xy[action] * self.step_size
      print("New gaze: ", self.gaze)

  def _create_action_space(self, num_actions):
    total_actions = num_actions + self.actions_end  # Gaze control
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
    imgs_undef = self.img_fov is None or self.img_periph is None
    gaze_changed = (self._last_observed_gaze != self.gaze).any()
    if imgs_undef or gaze_changed:
      # crop to fovea (centre region)
      h, w, ch = img.shape[0], img.shape[1], img.shape[2]

      pixels_h = int(h * self.fov_fraction)
      pixels_w = int(w * self.fov_fraction)

      # TODO: check index is within bounds
      self.img_fov = img[self.gaze[1]:self.gaze[1] + pixels_h, self.gaze[0]:self.gaze[0] + pixels_w, :]

      # zero foveal region, and return as peripheral (other options are to blur or cut it up differently)
      self.img_periph = img.copy()
      self.img_periph[self.gaze[1]:self.gaze[1] + pixels_h, self.gaze[0]:self.gaze[0] + pixels_w, :] = 0.

      self._last_observed_gaze = self.gaze.copy()

      # debugging
      # plt.imsave(str(self.i)+'_fov.png', self.img_fov)
      # plt.imsave(str(self.i)+'_periph.png', self.img_periph)
      # self.i += 1

    observation = {
      'full': img,
      'fovea': self.img_fov,
      'peripheral': self.img_periph}

    return observation
