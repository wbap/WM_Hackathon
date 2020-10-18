import logging
import sys
import numpy as np

from gym import error, spaces, utils
import gym_game
import pygame

import ray
import ray.tune as tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
import shutil

from ray.tune.registry import register_env
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import ray.rllib.agents.ppo as ppo
from ray.rllib.utils.framework import try_import_torch


torch, nn = try_import_torch()

# default config
config = {
  "mtl": {},
  "sc": {},
  "pfc": {}
}

from ray.rllib.models.preprocessors import Preprocessor
class StubPreprocessor(Preprocessor):
  """Test of a custom preprocessor - not required, for now, as this functionality is now in the wrapping Environment."""

  def __init__(self, obs_space, options):
    super().__init__(obs_space, options)

  def _init_shape(self, obs_space, options):
    print('Obs space:', str(obs_space))
    tx_shape = (10,10,10)
    #tx_shape = obs_space
    return tx_shape # can vary depending on inputs
  def transform(self, observation):
    tx = np.zeros((10,10,10))
    #tx = observation
    return tx  # return the preprocessed observation


def prefrontal_cortex(agent_action):
  pfc_action = agent_action
  print("======> StubAgent: agent_action", agent_action)
  return pfc_action


class StubAgent(TorchModelV2, nn.Module):
  """PyTorch custom model that flattens the input to 1d and delegates to a fc-net."""

  def __init__(self, obs_space, action_space, num_outputs, model_config, name):
    self.custom_model_config = config

    # Reshape obs to vector and convert to float
    volume = np.prod(obs_space.shape)
    space = np.zeros(volume)
    flat_observation_space = spaces.Box(low=0, high=255, shape=space.shape, dtype=np.float32)

    # TODO: Transform to output of any other PyTorch and pass new shape to model.

    # Create default model (for RL)
    TorchModelV2.__init__(self, flat_observation_space, action_space, num_outputs, model_config, name)
    nn.Module.__init__(self)
    self.torch_sub_model = TorchFC(flat_observation_space, action_space, num_outputs, model_config, name)

  def forward(self, input_dict, state, seq_lens):
    # flatten
    obs_4d = input_dict["obs"].float()
    volume = np.prod(obs_4d.shape[1:])  # calculate volume as vector excl. batch dim
    obs_3d_shape = [obs_4d.shape[0], volume]  # [batch size, volume]
    obs_3d = np.reshape(obs_4d, obs_3d_shape)
    input_dict["obs"] = obs_3d

    # print(input_dict["obs"])

    # TODO: forward() any other PyTorch modules here, pass result to RL algo
    # img_fov = obs["fovea"]
    # img_periph = obs["peripheral"]
    # fov_dog_pos, fov_dog_neg = self.retina.forward(img_fov)
    # periph_dog_pos, periph_dog_neg = self.retina.forward(img_periph)
    # what = self.vc_fovea.forward(fov_dog_pos, fov_dog_neg)
    # context = self.vc_periph.forward(periph_dog_pos, periph_dog_neg)
    #
    # gaze_pos = obs["gaze"]
    # where = self.pe.forward(gaze_pos)
    # what_where = (where, what, context)
    #
    # mtl_out = self.mtl.forward(what_where)
    #
    # ---> create circular buffer for mtl outputs
    # buffer = buffer.add(mtl_out)
    #
    # gaze_target, buffer = self.pfc.forward(gaze_target, buffer)
    #
    # 'buffer' goes to BG (the rllib agent)

    # Defer to default FC model
    fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)

    pfc_action = prefrontal_cortex(fc_out)

    return pfc_action, []

  def value_function(self):
    return torch.reshape(self.torch_sub_model.value_function(), [-1])

    # for name, module in model.named_children():
    #  if name in ['conv4', 'conv5']:
    #      print(module)