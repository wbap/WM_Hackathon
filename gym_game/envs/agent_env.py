import math
import json
from collections import deque
from timeit import default_timer as timer

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from abc import ABC, abstractmethod
import numpy as np
import pygame as pygame


from ray.rllib.utils.framework import try_import_torch

from agent.stubs.medial_temporal_lobe import MedialTemporalLobe
from agent.stubs.positional_encoder import PositionalEncoder
from agent.stubs.prefrontal_cortex import PrefrontalCortex
from agent.stubs.superior_colliculus import SuperiorColliculus
from agent.stubs.visual_path import VisualPath
from utils.general_utils import mergedicts

torch, nn = try_import_torch()


"""
  Wraps a task-specific environment and implements brain modules that are not trained by Reinforcement Learning.
"""


class DelayMessage:

  def __init__(self, delay_length):
    self.buffer = deque([], delay_length)

  def next_message(self, message_in):
    self.buffer.append(message_in)      # put a new item in the dequeue (left)
    message_out = self.buffer.pop()     # take one item off the deque (right)
    return message_out


def sc_2_env(sc_action):
  return sc_action


class AgentEnv(gym.Env):

  # Observation keys - for the obs dict that is emitted by this environment
  OBS_FOVEA = 'fovea'
  OBS_PERIPHERAL = 'peripheral'
  OBS_POSITIONAL_ENCODING = 'gaze'

  # module names (some modules use the corresponding obs key)
  MODULE_PFC = "pfc"
  MODULE_MTL = "mtl"
  MODULE_SC = "sc"

  @staticmethod
  def get_default_config():
    pe_config = PositionalEncoder.get_default_config()
    vp_f_config = VisualPath.get_default_config()
    vp_p_config = VisualPath.get_default_config()
    mtl_config = MedialTemporalLobe.get_default_config()
    sc_config = SuperiorColliculus.get_default_config()
    pfc_config = PrefrontalCortex.get_default_config()
    agent_config = {
      'obs_keys': {
        'visual': [AgentEnv.OBS_FOVEA, AgentEnv.OBS_PERIPHERAL]
      },
      AgentEnv.OBS_FOVEA: vp_f_config,
      AgentEnv.OBS_PERIPHERAL: vp_p_config,
      AgentEnv.OBS_POSITIONAL_ENCODING: pe_config,
      AgentEnv.MODULE_SC: sc_config,
      AgentEnv.MODULE_MTL: mtl_config,
      AgentEnv.MODULE_PFC: pfc_config,
      'mtl_input_delay_size': 1,
      'pfc_output_delay_size': 1
    }

    return agent_config

  @staticmethod
  def update_config(default_config, delta_config):
    """
    Override the config selectively. Return a complete config.
    """
    #updated_config = {**default_config, **delta_config}
    updated_config = dict(mergedicts(default_config, delta_config))
    return updated_config

  def __init__(self, env_type, env_config_file, config_file):
    self.env = gym.make(env_type, config_file=env_config_file)
    #self.env = env_type(env_config_file)
    self.action_space = self.env.action_space
    self.env_observation_space = self.env.observation_space
    self.reward = None

    # Build networks to preprocess the observation space
    default_config = self.get_default_config()  # TODO make this override
    with open(config_file) as json_file:
      delta_config = json.load(json_file)
      self._config = self.update_config(default_config, delta_config)

    print("=======================> CONFIG IS: ", self._config)

    # gather all obs keys
    self._obs_keys = []
    for obs_keys_key in self._config['obs_keys'].keys():
      for obs_key in self._config['obs_keys'][obs_keys_key]:
        self._obs_keys += obs_key

    self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build all the components, and add the observation spaces to obs_spaces_dict
    obs_spaces_dict = {}
    self.modules = {}

    # positional encoding
    self._use_pe = "pe" in self._config["obs_keys"] and self._config["obs_keys"]["pe"]
    if self._use_pe:
      self._build_positional_encoder(obs_spaces_dict)

    # visual processing - create a parietal cortex for fovea and periphery
    self._use_visual = "visual" in self._config["obs_keys"] and self._config["obs_keys"]["visual"]
    if self._use_visual:
      self._build_visual_paths(obs_spaces_dict)

    # build Prefrontal Cortex
    pfc = PrefrontalCortex(self.MODULE_PFC, self._config[self.MODULE_PFC]).to(self._device)
    self.modules[self.MODULE_PFC] = pfc

    # build delay on input to Medial Temporal Lobe, and output of observations to Agent
    self.mtl_input_buffer = DelayMessage(self._config["mtl_input_delay_size"])
    self.pfc_output_buffer = DelayMessage(self._config["pfc_output_delay_size"])

    # build Medial Temporal Lobe
    mtl = MedialTemporalLobe(self.MODULE_MTL, self._config[self.MODULE_MTL]).to(self._device)
    self.modules[self.MODULE_MTL] = mtl

    # build Superior Colliculus
    sc = SuperiorColliculus(self.MODULE_SC, self._config[self.MODULE_SC]).to(self._device)
    self.modules[self.MODULE_SC] = sc

    # the new observation space dict from the processed streams
    self.observation_space = spaces.Dict(obs_spaces_dict)

  def reset(self):
    obs = self.env.reset()
    return self.forward_observation(obs)

  # -------------------------------------- Building Regions --------------------------------------
  # ------------ Visual Streams

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
      visual_path = VisualPath(obs_key, input_shape, config, device=self._device).to(self._device)
      self.modules[obs_key] = visual_path

      output_shape = visual_path.get_output_shape()
      obs_shape = self.create_observation_shape_visual(output_shape)
      obs_space = self.create_observation_space_visual(obs_shape)
      obs_spaces_dict.update({obs_key: obs_space})

  # ------------ Positional Encoding
  @staticmethod
  def create_input_shape_pe(env_observation_space, env_obs_key):
    """ Convert from observation space to PyTorch tensor """
    env_obs = env_observation_space[env_obs_key]
    gaze_shape = env_obs.shape[0]   # 1 dimensional list, with length 2 for x and y
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
    pe = PositionalEncoder(obs_key, input_shape, config, max_xy=(screen_shape[0], screen_shape[1])).to(self._device)
    self.modules[obs_key] = pe

    output_shape = pe.get_output_shape()
    obs_shape = self.create_observation_shape_pe(output_shape)
    obs_space = self.create_observation_space_pe(obs_shape)
    obs_spaces_dict.update({obs_key: obs_space})

  # -----------------------------------------------------------------------------------------------

  def forward_observation(self, observation):

    # process foveal and peripheral visual path
    what_where_obs_dict = {}
    if self._use_visual:
      for obs_key in self._config["obs_keys"]["visual"]:
        visual_path = self.modules[obs_key]
        input_tensor = self.obs_to_tensor(observation, obs_key)
        encoding_tensor, decoding, target = visual_path.forward(input_tensor)
        self.tensor_to_obs(encoding_tensor, what_where_obs_dict, obs_key)

    # process positional encoding
    if self._use_pe:
      obs_key = self.OBS_POSITIONAL_ENCODING
      pe = self.modules[obs_key]
      input_tensor = self.obs_to_tensor(observation, obs_key)
      pe_output = pe.forward(input_tensor)
      self.tensor_to_obs(pe_output, what_where_obs_dict, obs_key)

    # process everything downstream of posterior cortex
    what_where_obs_dict_delayed = self.mtl_input_buffer.next_message(what_where_obs_dict)
    mtl = self.modules[self.MODULE_MTL]
    mtl_out = mtl.forward(what_where_obs_dict_delayed)

    pfc = self.modules[self.MODULE_PFC]
    env_obs_dict, _ = pfc.forward(what_where_obs_dict, mtl_out, bg_action=None)
    env_obs_dict_delayed = self.pfc_output_buffer.next_message(env_obs_dict)

    return env_obs_dict_delayed

  def forward_action(self, bg_action):

    pfc = self.modules[self.MODULE_PFC]
    _, pfc_action = pfc.forward(None, None, bg_action)

    sc = self.modules[self.MODULE_SC]
    sc_action = sc.forward(pfc_action)
    env_action = sc_2_env(sc_action)    # convert to game environment action space

    return env_action

  def tensor_to_obs(self, output, obs_dict, obs_key):
    #print('output is', output)
    obs = torch.squeeze(output).detach().cpu().numpy()  # remove batch dim, detach graph, convert numpy
    #print('!!!!!!!!!!!!!!!!!:',obs_key,' output tensor shape:', obs.shape)
    obs_dict[obs_key] = obs

  def obs_to_tensor(self, observation, obs_key):
    obs = torch.tensor(observation[obs_key])
    obs_b = torch.unsqueeze(obs, 0)  # insert batch dimension 0
    #print('!!!!!!!!!!!!!!!!!:',obs_key,' input tensor shape:', obs_b.shape)
    return obs_b.to(self._device)

  def get_config(self):
    """ return a dictionary of params """
    return self.env.get_config()

  def step(self, action):

    debug_observation = False
    debug_timing = False

    start = None
    if debug_timing:
      print('>>>>>>>>>>> Stub step')
      start = timer()

    # Update PFC with current action, which flow through to motor actions
    env_action = self.forward_action(action)

    # Update the game env, based on actions originating in PFC (and direct from Actor)
    [obs, self.reward, is_end_state, additional] = self.env.step(env_action)

    # Update agent brain with new observations
    tx_obs = self.forward_observation(obs)

    emit = [tx_obs, self.reward, is_end_state, additional]

    # The purpose of this section is to verify that valid observations are emitted.
    if debug_observation:
      print('Tx Obs keys ', tx_obs.keys())
      o = tx_obs['full']
      print('Obs Shape = ', o.shape)
      import hashlib
      m = hashlib.md5()
      m.update(o)
      h = m.hexdigest()
      print(' Hash = ', h)

      print('SA-ENV: OBS STATS: ')
      for key, val in tx_obs.items():
        print("\t{}: {}, {}, {}".format(key, val.shape, val.min(), val.max()))

    if debug_timing:
      end = timer()
      print('Step elapsed time: ', str(end - start))  # Time in seconds, e.g. 5.38091952400282

    return emit

  def get_screen_shape(self):
    return self.env.get_screen_shape()

  def get_random(self):
    """Return the PRNG for this game"""
    return self.env.get_random()

  def seed(self, seed=None):
    return self.env.seed(seed)

  def get_time(self):
    """Returns game time in milliseconds"""
    return self.env.get_time()

  def get_observation(self):
    print('>>>>>>>>>>> Stub get obs')
    obs = self.env.get_observation()
    tx_obs = self.forward_observation(obs)
    return tx_obs

  def render(self, mode='human', close=False):
    return self.env.render(mode, close)


