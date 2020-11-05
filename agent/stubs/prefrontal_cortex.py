import torch.nn as nn


class PrefrontalCortex(nn.Module):

  @staticmethod
  def get_default_config():
    config = {
    }
    return config

  def __init__(self, name, config):
    super().__init__()
    self._name = name
    self._config = config
    self._build()

  def _build(self):
    pass

  def forward(self, what_where_obs_dict, mtl_out, bg_action):
    pfc_action = bg_action
    pfc_observation = what_where_obs_dict
    # print("======> StubAgent: bg_action", bg_action)
    return pfc_observation, pfc_action
