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

    Based on code at: https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
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

      pe[key] = pe[key].unsqueeze(0)
      self.register_buffer('pe_' + key, pe[key])

    output_shape = [-1, self._config['dims']]
    return output_shape

  def forward(self, xy_tensor):
    x = xy_tensor[0][0]
    y = xy_tensor[0][1]
    pe_x = Variable(self.pe_x[x, :], requires_grad=False).cuda()
    pe_y = Variable(self.pe_y[y, :], requires_grad=False).cuda()
    return pe_x, pe_y

  def get_output_shape(self):
    return self._output_shape


