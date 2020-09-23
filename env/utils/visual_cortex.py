import torch.nn as nn
from utils import image_utils


class VisualCortex():

  config = {}

  def __init__(self, input_shape, config=None):
    super().__init__()

    self._input_shape = input_shape
    if config is None:
      self._config = VisualCortex.config
    else:
      self._config = config

    self._modules = []
    self._build()

  def _build(self):
    fovea = SparseAutoencoder(self._input_shape, self._config).to(device)
    self._modules['fovea'] = fovea

  def forward(self, x_fovea, x_peripheral, x_position):

    for m in self._modules:
    m = self._modules[module]
    return m.forward(x)

  def get_module(self, module):
    m = self._modules[module]
    return m

  def save(self, module, file_path):
    m = self._modules[module]
    torch.save(model.state_dict(), file_path)

  def load(self, module, file_path):
    m = self._modules[module]
    m.load_state_dict(torch.load(file_path))

  def eval(self, module=None):
    if module is None:
      for m in self._modules:
        m.eval()
    else:
      m = self._modules[module]
      m.eval()

  def train(self, module=None):
    if module is None:
      for m in self._modules:
        m.train()
    else:
      m = self._modules[module]
      m.train()
