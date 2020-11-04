import torch.nn as nn


class SuperiorColliculus(nn.Module):

  @staticmethod
  def get_default_config():
    config = {
    }
    return config

  def __init__(self, name, config):
    super().__init__()

    self._name = name
    self._config = config

  def _build(self):
    pass

  def forward(self, pfc_action):
    """
      pfc_action: command from PFC. Gaze target in 'action' space
      Return: gaze target in absolute coordinates (pixels in screen space)

      Currently, this is a 'pass-through" component.
      In the future, one may want to change the implementation e.g. progressively move toward the target
    """

    sc_action = pfc_action
    # print("======> StubAgentEnv: agent_action", sc_action)

    return sc_action

