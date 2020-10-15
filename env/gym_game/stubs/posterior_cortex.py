import torch.nn as nn
from cerenaut_pt_core.components.sparse_autoencoder import SparseAutoencoder

from .image_utils import *
from .positional_encoder import PositionalEncoder
from .retina import *


class PosteriorCortex(nn.Module):
  """
  Retinal coding, then Visual Cortex feature extraction, and positional encoding of the gaze.
  """

  # STREAM_FOVEA = 'fovea'
  # STREAM_PERIPHERAL = 'peripheral'
  # NUM_STREAMS = 2

  MODULE_RETINA = 'retina'
  MODULE_CORTEX = 'cortex'

  @staticmethod
  def get_default_config():  #input_config):

    retina_config = Retina.get_default_config()
    cortex_config = SparseAutoencoder.get_default_config()
    stream_config = {
      'load': None,
      'retina': retina_config,
      'cortex': cortex_config
    }

    # config = {}
    # for stream in input_config.keys():
    #   config[stream] = stream_config
    # return config
    return stream_config

  @staticmethod
  def update_config(default_config, delta_config):
    """
    Override the config selectively. Return a complete config.
    """
    #updated_config = {**default_config, **delta_config}
    updated_config = dict(mergedicts(default_config, delta_config))
    return updated_config

  def __init__(self, name, input_shape, config):
    super().__init__()

    self._name = name
    self._input_shape = input_shape
    self._config = config
    # if config is None:
    #   self._config = PosteriorCortex.get_default_config()  #input_config)
    # else:
    #   self._config = config

    """
    We have several sub-components for the DoG+/- encodings of fovea and peripheral vision
    Args:
    input_config: A dict of shapes for each input stream
    config: A dict containing a config dict for each stream's modules
    """

    # Build networks to preprocess the observation space
    # self._output_shapes = {}
    # streams = input_config.keys()
    # for stream in streams:
    #   input_shape = input_config[stream]
    print('>>>>>>>>>>>>>>>>>> ', self._name, 'posterior_input_shape: ', input_shape)
    retina_output_shape = self._build_retina(input_shape)
    print('>>>>>>>>>>>>>>>>>> ', self._name, 'retina_output_shape: ', retina_output_shape)
    cortex_output_shape = self._build_visual_cortex(retina_output_shape)
    print('>>>>>>>>>>>>>>>>>> ', self._name, 'cortex_output_shape: ', cortex_output_shape)

    #observation_shape = self.get_observation_shape(cortex_output_shape)
    #observation_space = self.get_observation_space(observation_shape)

    #self._output_shapes[stream] = cortex_output_shape
    #self._output_spaces[stream] = observation_space
    self._output_shape = cortex_output_shape

    # Option to reload a trained set of parameters
    if self._config['load'] is not None:
      cpkt_file = self._config['load']
      print('Loading parameters from checkpoint: ', cpkt_file)
      self.load(cpkt_file)

  def get_output_shape(self):
    return self._output_shape

  def _build_retina(self, input_shape):
    c = input_shape[1]
    h = input_shape[2]
    w = input_shape[3]
    config = self._config[self.MODULE_RETINA]
    module = Retina(c, config)
    module_name = self.get_module_name(self.MODULE_RETINA)
    self._modules[module_name] = module
    output_shape = module.get_output_shape(h, w)
    return output_shape

  def _build_visual_cortex(self, input_shape):
    config = self._config[self.MODULE_CORTEX]
    module = SparseAutoencoder(input_shape, config)  #.to(device)
    module_name = self.get_module_name(self.MODULE_CORTEX)
    self._modules[module_name] = module
    output_shape = module.get_encoding_shape()
    return output_shape

  def get_module_name(self, module):
    return module

  def forward(self, x):

    # forward retina coding
    module_name = self.get_module_name(self.MODULE_RETINA)
    module = self._modules[module_name]
    retina_output, r_p, r_n = module.forward(x)

    # forward cortex feature detection
    module_name = self.get_module_name(self.MODULE_CORTEX)
    module = self._modules[module_name]
    encoding, decoding = module.forward(retina_output)
    target = retina_output
    return encoding, decoding, target

  # def get_module(self, module):
  #   m = self._modules[module]
  #   return m

  def save(self, file_path):
    #m = self._modules[module]
    #torch.save(model.state_dict(), file_path)
    torch.save(self.state_dict(), file_path)

  def load(self, file_path):
    #m = self._modules[module]
    #m.load_state_dict(torch.load(file_path))
    self.load_state_dict(torch.load(file_path))

  # def eval(self, module=None):
  #   if module is None:  # eval all
  #     for m in self._modules:
  #       m.eval()
  #   else:
  #     m = self._modules[module]
  #     m.eval()

  # def train(self, module=None):
  #   if module is None:  # train all
  #     for m in self._modules:
  #       m.train()
  #   else:
  #     m = self._modules[module]
  #     m.train()
