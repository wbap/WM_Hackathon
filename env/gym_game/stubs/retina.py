import torch.nn as nn
from .image_utils import *

class Retina(nn.Module):

  config = {
    'f_size': 7,
    'f_sigma': 2.0,
    'f_k': 1.6  # approximates Laplacian of Gaussian
  }

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
    self._dog_filter_pos = get_dog_image_filter(channels=self.channels,
                                                            size=self._config['f_size'],
                                                            sigma=self._config['f_sigma'],
                                                            k=self._config['f_k'])

    self._dog_filter_neg = get_dog_image_filter(channels=self.channels,
                                                            size=self._config['f_size'],
                                                            sigma=self._config['f_sigma'],
                                                            k=self._config['f_k'],
                                                            invert=True)

  def forward(self, image_tensor):
    interest_pos = self._dog_filter_pos(image_tensor)
    interest_neg = self._dog_filter_neg(image_tensor)
    return interest_pos, interest_neg
