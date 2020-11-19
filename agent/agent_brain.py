from collections import deque
import math
from gym import spaces
import numpy as np

from ray.rllib.utils.framework import try_import_torch

from agent.stubs.medial_temporal_lobe import MedialTemporalLobe
from agent.stubs.positional_encoder import PositionalEncoder
from agent.stubs.prefrontal_cortex import PrefrontalCortex
from agent.stubs.superior_colliculus import SuperiorColliculus
from agent.stubs.visual_path import VisualPath

torch, nn = try_import_torch()


class DelayMessage:

  def __init__(self, delay_length):
    self.buffer = deque([], delay_length)

  def next_message(self, message_in):
    self.buffer.append(message_in)      # put a new item in the dequeue (left)
    message_out = self.buffer.pop()     # take one item off the deque (right)
    return message_out


class AgentBrain(nn.Module):

  """
  This class represents the bulk of the agent's brain.
  It contains all the regions that are _not_ trained with RL,
  which is just the Basal Ganglia, contained within StubAgent.
  """

  # Stub modules to instantiate. Some modules provide observations. For these, the name starts with OBS,
  # and these are also the keys for the obs dict that is emitted by this environment
  OBS_FOVEA = 'fovea'
  OBS_PERIPHERAL = 'peripheral'
  OBS_POSITIONAL_ENCODING = 'gaze'
  MODULE_PFC = "pfc"
  MODULE_MTL = "mtl"
  MODULE_SC = "sc"

  @staticmethod
  def get_default_config():
    config = {}     # use the config from StubAgentEnv
    return config

  def __init__(self, name, config, game_obs_space, obs_spaces_dict):
    """
    game_obs_space is the 'Game' Gym Environment observation space
    obs_space_dict is an empty dic, that this class populates with the 'Stub Brain' Gym Environment observations
    """
    super().__init__()
    self._name = name
    self._config = config
    self.env_observation_space = game_obs_space
    self._build(obs_spaces_dict)

  def _build(self, obs_spaces_dict):
    """
    Build all the components, and add the PyGame Environment observation spaces to obs_spaces_dict
    """

    # positional encoding
    self._use_pe = "pe" in self._config["obs_keys"] and self._config["obs_keys"]["pe"]
    if self._use_pe:
      self._build_positional_encoder(obs_spaces_dict)

    # visual processing - create a parietal cortex for fovea and periphery
    self._use_visual = "visual" in self._config["obs_keys"] and self._config["obs_keys"]["visual"]
    if self._use_visual:
      self._build_visual_paths(obs_spaces_dict)

    # build Prefrontal Cortex
    pfc = PrefrontalCortex(self.MODULE_PFC, self._config[self.MODULE_PFC])
    self.add_module(self.MODULE_PFC, pfc)

    # build delay on input to Medial Temporal Lobe, and output of observations to Agent
    self.mtl_input_buffer = DelayMessage(self._config["mtl_input_delay_size"])
    self.pfc_output_buffer = DelayMessage(self._config["pfc_output_delay_size"])

    # build Medial Temporal Lobe
    mtl = MedialTemporalLobe(self.MODULE_MTL, self._config[self.MODULE_MTL])
    self.add_module(self.MODULE_MTL, mtl)

    # build Superior Colliculus
    sc = SuperiorColliculus(self.MODULE_SC, self._config[self.MODULE_SC])
    self.add_module(self.MODULE_SC, sc)

  # -------------------------------------- Building Regions --------------------------------------

  # ------------ Visual Paths

  @staticmethod
  def create_input_shape_visual(env_observation_space, env_obs_key):
    """ Convert from observation space to PyTorch tensor """
    env_obs = env_observation_space[env_obs_key]
    c_in = env_obs.shape[0]
    h = env_obs.shape[1]
    w = env_obs.shape[2]
    input_shape = [-1, c_in, h, w]
    return input_shape

  @staticmethod
  def create_observation_space_visual(observation_shape):
    """ Convert from observation shape to space"""
    observation_space = spaces.Box(low=-math.inf, high=math.inf, shape=observation_shape, dtype=np.float32)
    return observation_space

  @staticmethod
  def create_observation_shape_visual(network_shape):
    """ Convert from PyTorch tensor to observation shape """
    b = network_shape[0]
    c = network_shape[1]
    h = network_shape[2]
    w = network_shape[3]
    observation_shape = [c, h, w]
    return observation_shape

  def _build_visual_paths(self, obs_spaces_dict):
    for obs_key in self._config["obs_keys"]["visual"]:
      input_shape = self.create_input_shape_visual(self.env_observation_space, obs_key)
      config = self._config[obs_key]
      visual_path = VisualPath(obs_key, input_shape, config)
      self.add_module(obs_key, visual_path)

      output_shape = visual_path.get_output_shape()
      obs_shape = self.create_observation_shape_visual(output_shape)
      obs_space = self.create_observation_space_visual(obs_shape)
      obs_spaces_dict.update({obs_key: obs_space})

  # ------------ Positional Encoding

  @staticmethod
  def create_input_shape_pe(env_observation_space, env_obs_key):
    """ Convert from observation space to PyTorch tensor """
    env_obs = env_observation_space[env_obs_key]
    gaze_shape = env_obs.shape[0]  # 1 dimensional list, with length 2 for x and y
    input_shape = [-1, gaze_shape]
    return input_shape

  @staticmethod
  def create_observation_space_pe(observation_shape):
    """ Convert from observation shape to space"""
    observation_space = spaces.Box(low=-math.inf, high=math.inf, shape=observation_shape, dtype=np.float32)
    return observation_space

  @staticmethod
  def create_observation_shape_pe(network_shape):
    """ Convert from PyTorch tensor to observation shape """
    b = network_shape[0]
    xy = network_shape[1]
    observation_shape = [xy]
    return observation_shape

  def _build_positional_encoder(self, obs_spaces_dict):
    obs_key = self.OBS_POSITIONAL_ENCODING
    input_shape = self.create_input_shape_pe(self.env_observation_space, obs_key)
    screen_shape = self.get_screen_shape()
    config = self._config[obs_key]
    pe = PositionalEncoder(obs_key, input_shape, config, max_xy=(screen_shape[0], screen_shape[1]))
    self.add_module(obs_key, pe)

    output_shape = pe.get_output_shape()
    obs_shape = self.create_observation_shape_pe(output_shape)
    obs_space = self.create_observation_space_pe(obs_shape)
    obs_spaces_dict.update({obs_key: obs_space})

  def forward(self, fwd_type, bg_action, observation_dic):
    """
    'Observation': a dictionary of Gym Env (the game environment) observations, converted to tensors
    'bg_action': an action from the Gym Agent (the basal ganglia)
    """

    if fwd_type == 'obs':

      # process foveal and peripheral visual path
      what_where_obs_dict = {}
      if self._use_visual:
        for obs_key in self._config["obs_keys"]["visual"]:
          visual_path = getattr(self, obs_key)
          input_tensor = observation_dic[obs_key]
          encoding_tensor, decoding, target = visual_path(input_tensor)
          what_where_obs_dict[obs_key] = encoding_tensor
          # self.tensor_to_obs(encoding_tensor, what_where_obs_dict, obs_key)

      # process positional encoding
      if self._use_pe:
        obs_key = self.OBS_POSITIONAL_ENCODING
        pe = getattr(self, obs_key)
        input_tensor = observation_dic[obs_key]
        pe_output = pe(input_tensor)
        what_where_obs_dict[obs_key] = pe_output

      # process everything downstream of posterior cortex
      what_where_obs_dict_delayed = self.mtl_input_buffer.next_message(what_where_obs_dict)
      mtl = getattr(self, self.MODULE_MTL)
      mtl_out = mtl(what_where_obs_dict_delayed)

      pfc = getattr(self, self.MODULE_PFC)
      pfc_obs_dict, _ = pfc(what_where_obs_dict, mtl_out, bg_action=None)
      pfc_obs_dict_delayed = self.pfc_output_buffer.next_message(pfc_obs_dict)

      return pfc_obs_dict_delayed

    elif fwd_type == 'action':

      pfc = getattr(self, self.MODULE_PFC)
      _, pfc_action = pfc(None, None, bg_action)

      sc = getattr(self, self.MODULE_SC)
      sc_action = sc(pfc_action)

      return sc_action
