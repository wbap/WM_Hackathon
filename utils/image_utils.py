from skimage.util import img_as_float32
import numpy as np


def fast_resize(image, scale):
  """
    image -- ndarray image as 8bit unsigned integers i.e. (0, 255)
    Return -- PIL image
  """

  from PIL import Image
  pili = Image.fromarray(image.astype('uint8'), 'RGB')
  size = (int(image.shape[0] * scale), int(image.shape[1] * scale))
  pili2 = pili.resize(size, Image.ANTIALIAS)
  return pili2


def to_pytorch_from_uint8(img):
  """
  Convert from uint8 images, with standard dimension order of [h,w,c]
  To PyTorch format, float32, dimension order of [c,h,w]

  """
  order = (2, 0, 1)
  img = img_as_float32(img)  # convert from uint8 to float 32
  img = np.transpose(img, order)
  return img
