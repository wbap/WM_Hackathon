import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn


# TODO: add appropriate padding so that the filtered image is the correct size


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
    weight = torch.reshape(weight,
                           [channels, 1, weight.shape[0], weight.shape[1]])  # [out_c, in_c/group, ksize[0], kszie[1]]
    self.register_buffer('weight', weight)

  def forward(self, input):
    """
    Apply filter to input.
    Arguments:
      input (torch.Tensor): Input to apply gaussian filter on.
    Returns:
      filtered (torch.Tensor): Filtered output.
    """
    print("input shape: ", input.shape)
    print("weight shape: ", self.weight.shape)

    return torch.conv2d(input, weight=self.weight, groups=self.groups)


def get_gaussian_image_filter(channels, size, sigma):
  kernel = gaussian_kernel(size, sigma)
  image_filter = ImageFilter(channels, kernel)
  return image_filter


def get_dog_image_filter(channels, size, sigma, k=1.6, invert=False):
  kernel = dog_kernel(size, sigma, k)
  if invert is True:
    kernel = -kernel
  image_filter = ImageFilter(channels, kernel)
  return image_filter
