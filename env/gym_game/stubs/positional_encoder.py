import math

import torch.nn as nn
import torch
from torch.autograd import Variable


class PositionalEncoder(nn.Module):

  @staticmethod
  def get_default_config():
    config = {
      'dims': 10
    }
    return config

  def __init__(self, name, input_shape, config, max_xy):
    super().__init__()

    self._name = name
    self._input_shape = input_shape
    self._config = config
    self._output_shape = self._build(max_xy)

  def _build(self, max_xy):
    """
    # input = gaze x, y
    # output = position

    Ideas from code at: https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
    """

    dims = self._config['dims']

    # create constant 'pe' matrix with values dependant on pos (pixel position) and number of dimensions (i)

    max_lengths = {'x': max_xy[0], 'y': max_xy[1]}
    pe = {}
    for key, mx in max_lengths.items():
      pe[key] = torch.zeros(mx, dims)
      for pos in range(mx):
        for i in range(0, dims, 2):
          pe[key][pos, i] = math.sin(pos / (10000 ** ((2 * i) / dims)))
          pe[key][pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / dims)))

      pe[key].unsqueeze(0)
      self.register_buffer('pe_' + key, pe[key])

    output_shape = [-1, 2 * self._config['dims']]   # x2 because of concatenation of x and y
    return output_shape

  def forward(self, xy_tensor):
    """ Take the appropriate slice (based on x and y) of the pre-computed positional encoding values """

    x = int(xy_tensor[0][0])
    y = int(xy_tensor[0][1])

    pe_x_copy = self.pe_x[x, :].clone().detach()  # ensure copy, not view
    pe_y_copy = self.pe_y[y, :].clone().detach()

    pe_xy = torch.cat([pe_x_copy, pe_y_copy])   # concatenate the x and y components
    pe_xy = torch.unsqueeze(pe_xy, 0)           # add in an empty batch dim

    return pe_xy

  def get_output_shape(self):
    return self._output_shape


