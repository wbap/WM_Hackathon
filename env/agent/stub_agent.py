import numpy as np

from gym import error, spaces, utils
import gym_game
import pygame

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


from utils.medial_temporal_lobe import MedialTemporalLobe
from utils.positional_encoding import PositionalEncoder
from utils.retina import Retina


torch, nn = try_import_torch()

# default config
config = {
  "retina": {
    'f_size': 7,
    'f_sigma': 2.0,
    'f_k': 1.6  # approximates Laplacian of Gaussian
  },
  "positional_encoding": {},
  "vc_fovea": {},
  "vc_periphery": {},
  "mtl": {},
  "sc": {}
}


class StubAgent(TorchModelV2, nn.Module):
  """PyTorch custom model that flattens the input to 1d and delegates to a fc-net."""

  def __init__(self, obs_space, action_space, num_outputs, model_config, name):
    # Reshape obs to vector and convert to float
    volume = np.prod(obs_space.shape)
    space = np.zeros(volume)
    flat_observation_space = spaces.Box(low=0, high=255, shape=space.shape, dtype=np.float32)

    # TODO: Transform to output of any other PyTorch and pass new shape to model.

    # Create default model (for RL)
    TorchModelV2.__init__(self, flat_observation_space, action_space, num_outputs, model_config, name)
    nn.Module.__init__(self)
    self.torch_sub_model = TorchFC(flat_observation_space, action_space, num_outputs, model_config, name)

    # create the stubbed sub-modules of the simple agent
    self._build()

  def _build(self):
    retina = Retina(1, config=None)
    self.add_module('retina', retina)

    pe_config = self.custom_model_config["positional_encoding"]
    pe = PositionalEncoder(config=pe_config)
    self.add_module('pe', pe)

    mtl_config = self.custom_model_config["mtl"]
    mtl = MedialTemporalLobe(config=mtl_config)
    self.add_module('mtl', mtl)

  def forward(self, input_dict, state, seq_lens):
    # flatten
    obs_4d = input_dict["obs"].float()
    volume = np.prod(obs_4d.shape[1:])  # calculate volume as vector excl. batch dim
    obs_3d_shape = [obs_4d.shape[0], volume]  # [batch size, volume]
    obs_3d = np.reshape(obs_4d, obs_3d_shape)
    input_dict["obs"] = obs_3d

    # TODO: forward() any other PyTorch modules here, pass result to RL algo
    # self.retina.forward()

    # Defer to default FC model
    fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
    return fc_out, []

  def value_function(self):
    return torch.reshape(self.torch_sub_model.value_function(), [-1])