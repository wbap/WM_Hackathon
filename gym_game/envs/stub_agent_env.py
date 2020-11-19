import json
from timeit import default_timer as timer

import gym
from gym import error, spaces, utils


from ray.rllib.utils.framework import try_import_torch

from agent.agent_brain import AgentBrain
from agent.stubs.medial_temporal_lobe import MedialTemporalLobe
from agent.stubs.positional_encoder import PositionalEncoder
from agent.stubs.prefrontal_cortex import PrefrontalCortex
from agent.stubs.superior_colliculus import SuperiorColliculus
from agent.stubs.visual_path import VisualPath, WriterSingleton
from utils.general_utils import mergedicts

torch, nn = try_import_torch()


"""
  Wraps a task-specific environment and implements brain modules that are not trained by Reinforcement Learning.
"""


def sc_2_env(sc_action):
  return sc_action


class StubAgentEnv(gym.Env):

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
        'visual': [AgentBrain.OBS_FOVEA, AgentBrain.OBS_PERIPHERAL]
      },
      AgentBrain.OBS_FOVEA: vp_f_config,
      AgentBrain.OBS_PERIPHERAL: vp_p_config,
      AgentBrain.OBS_POSITIONAL_ENCODING: pe_config,
      AgentBrain.MODULE_SC: sc_config,
      AgentBrain.MODULE_MTL: mtl_config,
      AgentBrain.MODULE_PFC: pfc_config,
      'mtl_input_delay_size': 1,
      'pfc_output_delay_size': 1
    }

    return agent_config

  @staticmethod
  def update_config(default_config, delta_config):
    """
    Override the config selectively. Return a complete config.
    """
    updated_config = dict(mergedicts(default_config, delta_config))
    return updated_config

  def __init__(self, env_type, env_config_file, config_file):
    self.env = gym.make(env_type, config_file=env_config_file)
    self.action_space = self.env.action_space
    self.env_observation_space = self.env.observation_space
    self.reward = None

    # Build networks to preprocess the observation space
    default_config = self.get_default_config()  # TODO make this override
    with open(config_file) as json_file:
      delta_config = json.load(json_file)
      self._config = self.update_config(default_config, delta_config)

    print("=======================> CONFIG IS: ", self._config)

    # set the device to gpu is possible
    self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # instantiate the portion of the brain that is not trained with RL (and therefore located here in env)
    obs_spaces_dict = {}
    self.agent_brain = AgentBrain('brain', self._config, self.env_observation_space, obs_spaces_dict).to(self._device)

    # the new observation space dict from the processed streams
    self.observation_space = spaces.Dict(obs_spaces_dict)

  def reset(self):
    game_obs_dic = self.env.reset()
    game_obs_dic_tensors = {obs_key: self.obs_to_tensor(obs) for (obs_key, obs) in game_obs_dic.items()}
    obs_dic_tensors = self.agent_brain(fwd_type='obs', bg_action=None, observation_dic=game_obs_dic_tensors)
    obs_dic = {obs_key: self.tensor_to_obs(t_obs) for (obs_key, t_obs) in obs_dic_tensors.items()}
    return obs_dic

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
    sc_action = self.agent_brain(fwd_type='action', bg_action=action, observation_dic=None)
    env_action = sc_2_env(sc_action)  # convert to game environment action space

    # Update the game env, based on actions originating in PFC (and direct from Actor)
    [game_obs_dic, self.reward, is_end_state, additional] = self.env.step(env_action)

    # Update agent brain with new observations
    game_obs_dic_tensors = {obs_key: self.obs_to_tensor(obs) for (obs_key, obs) in game_obs_dic.items()}
    obs_dic_tensors = self.agent_brain(fwd_type='obs', bg_action=None, observation_dic=game_obs_dic_tensors)
    obs_dic = {obs_key: self.tensor_to_obs(t_obs) for (obs_key, t_obs) in obs_dic_tensors.items()}

    emit = [obs_dic, self.reward, is_end_state, additional]

    # The purpose of this section is to verify that valid observations are emitted.
    if debug_observation:
      print('Tx Obs keys ', obs_dic.keys())
      o = obs_dic['full']
      print('Obs Shape = ', o.shape)
      import hashlib
      m = hashlib.md5()
      m.update(o)
      h = m.hexdigest()
      print(' Hash = ', h)

      print('SA-ENV: OBS STATS: ')
      for key, val in obs_dic.items():
        print("\t{}: {}, {}, {}".format(key, val.shape, val.min(), val.max()))

    if debug_timing:
      end = timer()
      print('Step elapsed time: ', str(end - start))  # Time in seconds, e.g. 5.38091952400282

    # print("-------------------------- graph ---------------------------")
    # writer = WriterSingleton.get_writer()
    # writer.add_graph(model=self.agent_brain, input_to_model=('obs', action, game_obs_dic_tensors), verbose=True)
    # writer.flush()
    
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
    tx_obs = self.agent_brain('obs', obs, None)
    return tx_obs

  def render(self, mode='human', close=False):
    return self.env.render(mode, close)

  def obs_to_tensor(self, obs):
    tensor_obs = torch.tensor(obs)
    tensor_obs_b = torch.unsqueeze(tensor_obs, 0)  # insert batch dimension 0
    #print('!!!!!!!!!!!!!!!!!:',obs_key,' input tensor shape:', obs_b.shape)
    return tensor_obs_b.to(self._device)

  @staticmethod
  def tensor_to_obs(tensor_obs):
    #print('output is', output)
    obs = torch.squeeze(tensor_obs).detach().cpu().numpy()  # remove batch dim, detach graph, convert numpy
    #print('!!!!!!!!!!!!!!!!!:',obs_key,' output tensor shape:', obs.shape)
    return obs
