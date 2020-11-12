import torch.nn as nn
import torch
import torchvision

from utils.image_filter_utils import get_dog_image_filter, conv2d_output_shape
from utils.writer_singleton import WriterSingleton


class Retina(nn.Module):
  STEP = 0

  @staticmethod
  def get_default_config():
    config = {
      'f_size': 7,
      'f_sigma': 2.0,
      'f_k': 1.6  # approximates Laplacian of Gaussian
    }
    return config    

  def __init__(self, name, channels, config=None, device=None):
    super().__init__()

    self._name = name
    self.channels = channels
    if config is None:
      self._config = Retina.config
    else:
      self._config = config

    self.summaries = self._config['summaries']

    self._device = device
    self._dog_filter_pos = None
    self._dog_filter_neg = None

    self._build()

  def _build(self):
    # DoG kernel - edge and corner detection plus smoothing
    size = self._config['f_size']
    sigma = self._config['f_sigma']
    k = self._config['f_k']
    self._dog_filter_pos = get_dog_image_filter(channels=self.channels, size=size, sigma=sigma,
                                                device=self._device, k=k)
    self._dog_filter_neg = get_dog_image_filter(channels=self.channels, size=size, sigma=sigma,
                                                device=self._device, k=k, invert=True)

  def forward(self, image_tensor):
    interest_pos = self._dog_filter_pos(image_tensor)
    interest_neg = self._dog_filter_neg(image_tensor)
    channel_dim = 1  # B,C,H,W
    interest = torch.cat([interest_pos, interest_neg], dim=channel_dim)

    writer = WriterSingleton.get_writer()
    if self.summaries and writer:
      self.STEP += 1
      # print("retina/input shape: ", image_tensor.shape)
      # print("retina/dog_neg shape: ", interest_neg.shape)
      writer.add_image(self._name + '/input', torchvision.utils.make_grid(image_tensor), global_step=self.STEP)
      writer.add_image(self._name + '/dog-', torchvision.utils.make_grid(interest_neg), global_step=self.STEP)
      writer.add_image(self._name + '/dog+', torchvision.utils.make_grid(interest_pos), global_step=self.STEP)

      writer.add_histogram(self._name + '/hist-input', image_tensor, global_step=self.STEP)
      writer.add_histogram(self._name + '/hist-dog-', interest_neg, global_step=self.STEP)
      writer.add_histogram(self._name + '/hist-dog+', interest_pos, global_step=self.STEP)
      writer.flush()

    return interest, interest_pos, interest_neg

  def get_output_size(self, h, w):
    kernel_size = self._config['f_size']
    output_shape = conv2d_output_shape([h,w], kernel_size=kernel_size, stride=1, pad=0, dilation=1)
    return output_shape

  def get_output_shape(self, h, w):
    output_size = self.get_output_size(h, w)
    output_shape = [-1, self.channels * 2, output_size[0], output_size[1]]  # because 2x 3 channels (+/-)
    return output_shape
