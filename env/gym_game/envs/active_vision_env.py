import logging

import numpy as np
from gym import spaces
from skimage.transform import rescale

from .pygame_env import PyGameEnv

"""
  ActiveVisionEnv receives a screen image.
  All other quantities are expressed relative to that screen image. 
"""


class ActiveVisionEnv(PyGameEnv):
  metadata = {'render.modes': ['human', 'rgb_array']}

  HUMAN = 'human'
  ARRAY = 'rgb_array'

  def __init__(self, num_actions, screen_width, screen_height, frame_rate):
    """
    The concrete environment (e.g. dm2s-v0) is responsible for reading in the config in its own format
    and creating a dictionary of params before calling super.init() to get to here.
    i.e. the params are available via self.get_config()
    """
    config = self.get_config()
    self.fov_fraction = float(config["fovea_fraction"])  # fovea occupies this fraction of the screen image (applied to x and y respectively)
    self.fov_scale = float(config["fovea_scale"])  # image size, expressed as fraction of screen size
    self.step_size = int(config["gaze_step_size"])  # step size of gaze movement, in pixels in the screen image
    self.peripheral_scale = float(config["peripheral_scale"])  # peripheral image size, expressed as fraction of screen size
    self.screen_scale = float(config["screen_scale"])  # resize the screen image before returning as an observation
    self.fov_size = np.array([int(self.fov_fraction * screen_width), int(self.fov_fraction * screen_height)])
    self.gaze = np.array([screen_width // 2, screen_height // 2])  # gaze position - (x, y)--> *top left of fovea*
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

    self._x_min = self.fov_size[0]
    self._x_max = screen_width - self.fov_size[0]
    self._y_min = self.fov_size[1]
    self._y_max = screen_height - self.fov_size[1]
    # self.i = 0      # used for debugging

    super().__init__(num_actions, screen_width, screen_height, frame_rate)

  def reset(self):
    """Reset gaze coordinates"""
    return super().reset()

  def _do_step(self, action, time):
    # update the position of the fovea (fov_pos), given the action taken
    logging.debug("Received action: ", action)
    # if within action scope, modify gaze
    if self._actions_start <= action < self._actions_end:
      self.gaze = self.gaze + self._action_2_xy[action] * self.step_size

      self.gaze[0] = np.clip(self.gaze[0], self._x_min, self._x_max)   # ensure x coord is in bounds
      self.gaze[1] = np.clip(self.gaze[1], self._y_min, self._y_max)   # ensure y coord is in bounds

      #print("New gaze: ", self.gaze)

  def _create_action_space(self, num_actions):
    total_actions = num_actions + self._actions_end  # Gaze control
    self.action_space = spaces.Discrete(total_actions)

  def _create_observation_space(self, screen_width, screen_height, channels=3):
    full_shape = self.get_full_observation_shape()
    fovea_shape = self.get_fovea_observation_shape()  #(channels, screen_height, screen_width)
    peripheral_shape = self.get_peripheral_observation_shape()  #(channels, screen_height, screen_width)
    full = spaces.Box(low=0, high=255, shape=full_shape, dtype=np.uint8)
    fovea = spaces.Box(low=0, high=1.0, shape=fovea_shape, dtype=np.float32)
    peripheral = spaces.Box(low=0, high=1.0, shape=peripheral_shape, dtype=np.float32)
    self.observation_space = spaces.Dict({
      'full': full,
      'fovea': fovea,
      'peripheral': peripheral})

  def get_full_observation_shape(self):
    h = self.screen_shape[0]
    w = self.screen_shape[1]
    c = self.screen_shape[2]
    h2 = int(h * self.screen_scale)
    w2 = int(w * self.screen_scale)
    return (c, h2, w2)

  def get_fovea_observation_shape(self):
    h = self.screen_shape[0]
    w = self.screen_shape[1]
    c = self.screen_shape[2]
    pixels_h = int(h * self.fov_fraction)
    pixels_w = int(w * self.fov_fraction)
    h2 = int(pixels_h * self.fov_scale)
    w2 = int(pixels_w * self.fov_scale)
    return (c, h2, w2)

  def get_peripheral_observation_shape(self):
    h = self.screen_shape[0]
    w = self.screen_shape[1]
    c = self.screen_shape[2]
    h2 = int(h * self.peripheral_scale)
    w2 = int(w * self.peripheral_scale)
    return (c, h2, w2)

  def get_observation(self):
    """
    Return images in PyTorch format.
    """

    debug = False
    multichannel = True

    img = self.render(mode='rgb_array')
    img = np.transpose(img, [1, 0, 2])  # observed img is horizontally reflected, and rotated 90 deg ...
    img_shape = img.shape

    # convert to float type (the standard for pytorch and other scikit image methods)
    from skimage.util import img_as_float
    img = img_as_float(img)

    # Peripheral Image - downsize to get peripheral (lower resolution) image
    self._img_periph = rescale(img, self.peripheral_scale, multichannel=multichannel)

    # Foveal Image - crop to fovea and rescale
    h, w, ch = img.shape[0], img.shape[1], img.shape[2]
    pixels_h = int(h * self.fov_fraction)
    pixels_w = int(w * self.fov_fraction)
    self._img_fov = img[self.gaze[1]:self.gaze[1] + pixels_h, self.gaze[0]:self.gaze[0] + pixels_w, :]
    self._img_fov = rescale(self._img_fov, self.fov_scale, multichannel=multichannel)

    # resize screen image before returning as observation
    img = rescale(img, self.screen_scale, multichannel=multichannel)

    # debugging
    if debug:
      print('img orig screen shape:', img_shape)
      print('img periph shape:', self._img_periph.shape)
      print('img fovea shape:', self._img_fov.shape)
      print('img screen rescaled shape:', img.shape)

      # import matplotlib.pyplot as plt
      # plt.imsave(str(self.i)+'_fov.png', self._img_fov)
      # plt.imsave(str(self.i)+'_periph.png', self._img_periph)
      # self.i += 1

    # PyTorch expects dimension order [b,c,h,w]
    # transpose dimensions from [,h,w,c] to [,c,h,w]]
    order = (2, 0, 1)
    self._img_full = np.transpose(img, order)
    self._img_fov = np.transpose(self._img_fov, order)
    self._img_periph = np.transpose(self._img_periph, order)
    # print('fovea shape trans:', self._img_fov.shape)

    # Assemble dict
    observation = {
      'full': self._img_full.astype(np.float32),#img.astype(np.float32),
      'fovea': self._img_fov.astype(np.float32),
      'peripheral': self._img_periph.astype(np.float32)}

    return observation
