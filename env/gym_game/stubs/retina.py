import torch.nn as nn
from .image_utils import *


class Retina(nn.Module):

  @staticmethod
  def get_default_config():
    config = {
      'f_size': 7,
      'f_sigma': 2.0,
      'f_k': 1.6  # approximates Laplacian of Gaussian
    }
    return config    

  def __init__(self, channels, config=None):
    super().__init__()

    self.channels = channels
    if config is None:
      self._config = Retina.config
    else:
      self._config = config

    # self._image_dic = {}
    self._dog_filter_pos = None
    self._dog_filter_neg = None

    self._build()

  # def set_image(self, name, val):
  #   self._image_dic[name] = val
  #
  # def get_image(self, name):
  #   return self._image_dic[name]

  def _build(self):
    # DoG kernel - edge and corner detection plus smoothing
    size = self._config['f_size']
    sigma = self._config['f_sigma']
    k = self._config['f_k']
    self._dog_filter_pos = get_dog_image_filter(channels=self.channels, size=size, sigma=sigma, k=k)
    self._dog_filter_neg = get_dog_image_filter(channels=self.channels, size=size, sigma=sigma, k=k, invert=True)

  def forward(self, image_tensor):
    interest_pos = self._dog_filter_pos(image_tensor)
    interest_neg = self._dog_filter_neg(image_tensor)
    channel_dim = 1  # B,C,H,W
    interest = torch.cat([interest_pos, interest_neg], dim=channel_dim)
    #return interest_pos, interest_neg
    return interest, interest_pos, interest_neg

  def get_output_size(self, h, w):
    kernel_size = self._config['f_size']
    output_shape = conv2d_output_shape([h,w], kernel_size=kernel_size, stride=1, pad=0, dilation=1)
    return output_shape

  def get_output_shape(self, h, w):
    output_size = self.get_output_size(h, w)
    output_shape = [-1, self.channels * 2, output_size[0], output_size[1]] # because 2x 3 channels (+/-)
    return output_shape