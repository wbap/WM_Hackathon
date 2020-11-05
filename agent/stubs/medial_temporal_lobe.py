from collections import deque
from typing import Any

import torch.nn as nn


class MedialTemporalLobe(nn.Module):

  @staticmethod
  def get_default_config():
    config = {
      "mtl_max_length": 1
    }
    return config

  def __init__(self, name, config):
    super().__init__()
    self._name = name
    self._config = config
    self.mtl = None
    self._build()

  def _build(self):
    self.mtl = deque([], self._config["mtl_max_length"])

  def forward(self, what_where):
    self.mtl.append(what_where)



