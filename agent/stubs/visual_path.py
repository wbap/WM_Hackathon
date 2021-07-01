import torch.nn as nn
from cerenaut_pt_core.components.sparse_autoencoder import SparseAutoencoder

from utils.general_utils import mergedicts
from .retina import *


class VisualPath(nn.Module):
  """
  Retinal coding, then Visual Cortex feature extraction, and positional encoding of the gaze.
  """

  MODULE_RETINA = 'retina'
  MODULE_CORTEX = 'cortex'
  STEP = 0

  @staticmethod
  def get_default_config():
    retina_config = Retina.get_default_config()
    cortex_config = SparseAutoencoder.get_default_config()
    stream_config = {
      'load': None,
      'retina': retina_config,
      'cortex': cortex_config
    }

    return stream_config

  @staticmethod
  def update_config(default_config, delta_config):
    """
    Override the config selectively. Return a complete config.
    """
    updated_config = dict(mergedicts(default_config, delta_config))
    return updated_config

  def __init__(self, name, input_shape, config, device):
    """
    We have several sub-components for the DoG+/- encodings of fovea and peripheral vision
    Args:
    input_config: A dict of shapes for each input stream
    config: A dict containing a config dict for each stream's modules
    """

    super().__init__()

    self._name = name
    self._input_shape = input_shape
    self._config = config
    self._device = device
    self.summaries = True
    
    # Build networks to preprocess the observation space
    print('>>>>>>>>>>>>>>>>>> ', self._name, 'visual_cortex_input_shape: ', input_shape)
    retina_output_shape = self._build_retina(input_shape)
    print('>>>>>>>>>>>>>>>>>> ', self._name, 'retina_output_shape: ', retina_output_shape)
    cortex_output_shape = self._build_visual_cortex(retina_output_shape)
    print('>>>>>>>>>>>>>>>>>> ', self._name, 'visual_cortex_output_shape: ', cortex_output_shape)

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
    module = Retina(self._name + '/retina', c, config, self._device)
    module_name = self.get_module_name(self.MODULE_RETINA)
    self._modules[module_name] = module
    output_shape = module.get_output_shape(h, w)
    return output_shape

  def _build_visual_cortex(self, input_shape):
    config = self._config[self.MODULE_CORTEX]
    module = SparseAutoencoder(input_shape, config)
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

    if self.summaries:
      self.STEP += 1
      writer = WriterSingleton.get_writer()
      writer.add_histogram(self._name + '/retina-input', x, global_step=self.STEP)
      writer.add_histogram(self._name + '/retina-output', retina_output, global_step=self.STEP)

      dog_pos = retina_output[:, 0:3, :, :]
      dog_neg = retina_output[:, 3:6, :, :]
      writer.add_image(self._name + '/retina-output-dog+', torchvision.utils.make_grid(dog_pos), global_step=self.STEP)
      writer.add_image(self._name + '/retina-output-dog-', torchvision.utils.make_grid(dog_neg), global_step=self.STEP)

      writer.flush()

    # forward cortex feature detection
    module_name = self.get_module_name(self.MODULE_CORTEX)
    module = self._modules[module_name]
    encoding, decoding = module.forward(retina_output)
    target = retina_output
    return encoding, decoding, target

  def save(self, file_path):
    torch.save(self.state_dict(), file_path)

  def load(self, file_path):
    #m = self._modules[module]
    #m.load_state_dict(torch.load(file_path))
    map_location = None
    if torch.cuda.is_available() is False:
      map_location = torch.device('cpu')

    self.load_state_dict(torch.load(file_path, map_location=map_location))

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
