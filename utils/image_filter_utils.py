import math

import numpy as np
import scipy.ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: add appropriate padding so that the filtered image is the correct size
def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
  """
  Utility function for computing output of convolutions
  takes a tuple of (h,w) and returns a tuple of (h,w)
  """
  from math import floor
  if type(kernel_size) is not tuple:
      kernel_size = (kernel_size, kernel_size)
  h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1)/stride) + 1)
  w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1)/stride) + 1)
  return h, w


def gaussian_kernel(size, stddev):
  if size % 2 == 0:
    raise RuntimeWarning("Kernel size must be odd, or the kernel is asymmetric")

  n = np.zeros((size, size))
  mid = (size // 2)
  n[mid, mid] = 1
  kernel = scipy.ndimage.gaussian_filter(n, sigma=stddev)
  return kernel


def dog_kernel(size, stddev, k=1.6):
  """
  If k is not defined, use the convention for the Laplacian of Gaussians, which is 1.6
  https://en.wikipedia.org/wiki/Difference_of_Gaussians
  """
  g1 = gaussian_kernel(size, stddev)
  g2 = gaussian_kernel(size, k * stddev)
  dog = g1 - g2
  return dog


class ImageFilter(nn.Module):
  """
  Apply an image filter WITHOUT training weights.
  Filtering is performed separately for each channel in the input using a depthwise convolution.

  Arguments:
      channels (int, sequence): Number of channels of the input tensors. Output will
          have this number of channels as well.
  """

  def __init__(self, channels, kernel):
    """
    Arguments:
      channels: number of channels in the image
      kernel (ndarray): for filtering operation
    """
    super(ImageFilter, self).__init__()

    self.groups = channels

    weight = torch.from_numpy(kernel).float()
    shape_4d1 = [1, 1, weight.shape[0], weight.shape[1]]
    weight = torch.reshape(weight, shape_4d1)
    #shape_4d = [channels, channels, weight.shape[0], weight.shape[1]]
    #weight = torch.reshape(weight, shape_4d)  # [out_c, in_c/group, ksize[0], kszie[1]]
    weight = weight.repeat([channels, channels // self.groups, 1, 1])
    #print('ImageFilter weight shape', weight.shape)
    self.register_buffer('weight', weight)

  def forward(self, input):
    """
    Apply filter to input.
    Arguments:
      input (torch.Tensor): Input to apply gaussian filter on.
    Returns:
      filtered (torch.Tensor): Filtered output.
    """
    #print("ImageFilter f() input shape: ", input.shape)
    #print("ImageFilter f() weight shape: ", self.weight.shape)

    # conv2d args:
    # input = [b,c,h,w]
    # weight = [c_out, c_in, kH, kW]
    # groups = 1 by default; split into groups. c_in should be divisible by groups
    # stride = not used
    return F.conv2d(input, weight=self.weight, groups=self.groups)


def get_gaussian_image_filter(channels, size, sigma, device):
  kernel = gaussian_kernel(size, sigma)
  image_filter = ImageFilter(channels, kernel).to(device)
  return image_filter


def get_dog_image_filter(channels, size, sigma, device, k=1.6, invert=False):
  kernel = dog_kernel(size, sigma, k)
  if invert is True:
    kernel = -kernel
  image_filter = ImageFilter(channels, kernel).to(device)
  return image_filter
