"""Model pretraining."""

from __future__ import print_function

import json
import argparse
import gym
import json
import shutil
import sys

import ray
import ray.rllib.agents.a3c as a3c
import ray.tune as tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets, transforms


from agent.stubs.visual_path import VisualPath, WriterSingleton
from gym_game.envs.pygame_dataset import PyGameDataset

show_encode_and_decode = True


def train(args, model, device, train_loader, global_step, optimizer, epoch, writer):
  """Trains the model for one epoch."""
  model.train()

  for batch_idx, (data) in enumerate(train_loader):
    #print('Data:', data)
    #print('Batch:', batch_idx)

    data = data.to(device)

    optimizer.zero_grad()
    encoding, output, target = model(data)
    # print('input min/max=', data.min(), data.max())
    # print('encoding shape', encoding.shape)
    # print('Decoding shape', output.shape)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()

    writer.add_image('pre-train/inputs', torchvision.utils.make_grid(data), global_step)
    writer.add_scalar('pre-train/loss', loss, global_step)

    # This section is for extra fine grained debugging and makes some assumptions about size and dimensions
    if show_encode_and_decode:
      import numpy as np
      encoding_volume = encoding.shape[1] * encoding.shape[2] * encoding.shape[3]
      side_length = int(np.sqrt(encoding_volume))     # TODO NOTE this assumes it is evenly square
      encoding_img = torch.reshape(encoding, [encoding.shape[0], 1, side_length, side_length])

      writer.add_image('pre-train/encoding', torchvision.utils.make_grid(encoding_img), global_step)

      # when input has 6 channels...
      dog_pos = output[:, 0:3, :, :]
      dog_neg = output[:, 3:6, :, :]
      writer.add_image('pre-train/dog+recon', torchvision.utils.make_grid(dog_pos), global_step)
      writer.add_image('pre-train/dog-recon', torchvision.utils.make_grid(dog_neg), global_step)

      writer.add_histogram('pre-train/hist-dog+recon', dog_pos, global_step=global_step)
      writer.add_histogram('pre-train/hist-dog-recon', dog_neg, global_step=global_step)
      writer.add_histogram('pre-train/hist-encoding', encoding, global_step=global_step)

    global_step += 1

    if batch_idx % args.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))

      if args.dry_run:
        break

  return global_step  


def test(model, device, test_loader, global_step, writer):
  """Evaluates the trained model."""
  model.eval()
  test_loss = 0

  with torch.no_grad():
    for batch_idx, (data) in enumerate(test_loader):
      data = data.to(device)
      encoding, output, target = model(data)

      writer.add_image('pre-test/inputs', torchvision.utils.make_grid(data), global_step)

      # This section is for extra fine grained debugging and makes some assumptions about size and dimensions
      if show_encode_and_decode:
        import numpy as np
        encoding_volume = encoding.shape[1] * encoding.shape[2] * encoding.shape[3]
        side_length = int(np.sqrt(encoding_volume))  # TODO NOTE this assumes it is evenly square
        encoding_img = torch.reshape(encoding, [encoding.shape[0], 1, side_length, side_length])

        writer.add_image('pre-test/encoding', torchvision.utils.make_grid(encoding_img), global_step)

        # when input has 6 channels...
        dog_pos = output[:, 0:2, :, :]
        dog_neg = output[:, 3:5, :, :]
        writer.add_image('pre-test/dog+recon', torchvision.utils.make_grid(dog_pos), global_step)
        writer.add_image('pre-test/dog-recon', torchvision.utils.make_grid(dog_neg), global_step)

        writer.add_histogram('pre-test/hist-dog+recon', dog_pos, global_step=global_step)
        writer.add_histogram('pre-test/hist-dog-recon', dog_neg, global_step=global_step)
        writer.add_histogram('pre-test/hist-encoding', encoding, global_step=global_step)

      test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    writer.add_scalar('pre-test/avg_loss', test_loss, global_step)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


def main():
  # Training settings
  parser = argparse.ArgumentParser(description='Module pretraining')
  parser.add_argument('--env', type=str, default='', metavar='N',
                      help='Gym environment name')
  parser.add_argument('--env-config', type=str, default='', metavar='N',
                      help='Gym environment config file')
  parser.add_argument('--env-data-dir', type=str, default='', metavar='N',
                      help='Gym environment pre-generated data directory')
  parser.add_argument('--env-obs-key', type=str, default=None, metavar='N',
                      help='Gym environment dict observation object key')
  parser.add_argument('--config', type=str, default='configs/pretrain.json', metavar='N',
                      help='Model configuration (default: configs/pretrain.json')
  parser.add_argument('--epochs', type=int, default=1, metavar='N',
                      help='Number of training epochs (default: 1)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--dry-run', action='store_true', default=False,
                      help='quickly check a single pass')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--model-file', type=str, default=None, metavar='N',
                      help='Trained model parameters file')

  args = parser.parse_args()

  torch.manual_seed(args.seed)

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  # Ensure output folder exists
  import os
  file_path = args.model_file
  dirname = os.path.dirname(file_path)
  if not os.path.exists(dirname):
    os.mkdir(dirname)

  # Read global config file
  with open(args.config) as config_file:
    config = json.load(config_file)

  # Obtain the Optimizer config
  optimizer_config = config['optimizer']
  batch_size = optimizer_config['batch_size']

  kwargs = {'batch_size': batch_size}
  if use_cuda:
    kwargs.update({
        'num_workers': 1,
        'pin_memory': True,
        'shuffle': True
    })

  writer = WriterSingleton.get_writer()
  transform = transforms.Compose([
      transforms.ToTensor()
  ])

  # Build environment
  env_name = args.env
  print('Making Gym[PyGame] environment:', env_name)
  env_config_file = args.env_config
  print('Env config file:', env_config_file)
  env = gym.make(env_name, config_file=env_config_file)
  print('Env constructed')

  # We will pretrain just from one observation at a time
  obs_key = args.env_obs_key
  print('Obs. key:', obs_key)
  dataset = PyGameDataset(key=obs_key)
  print('Loading pre-generated data from: ', args.env_data_dir) 
  read_ok = dataset.read(args.env_data_dir)
  print('Loaded pre-generated data?', str(read_ok)) 

  env.reset()
  data_shape = dataset.get_shape(env)
  print('Data shape:', data_shape)

  train_loader = torch.utils.data.DataLoader(dataset, **kwargs)
  test_loader = torch.utils.data.DataLoader(dataset, **kwargs)

  input_shape = (-1,) + data_shape  #[-1, 1, 28, 28]
  print('Final dataset shape:', input_shape)

  # Override model config
  default_model_config = VisualPath.get_default_config()
  delta_model_config = config['model']
  model_config = VisualPath.update_config(default_model_config, delta_model_config)
  print('Model config:\n', model_config)
  model = VisualPath(obs_key, input_shape, model_config, device=device).to(device)
  print('Model:', model)

  # Create optimizer
  optimizer = optim.Adam(model.parameters(), lr=optimizer_config['learning_rate'])

  # Begin training
  global_step = 0
  WriterSingleton.global_step = global_step
  for epoch in range(0, args.epochs):
    global_step = train(args, model, device, train_loader, global_step, optimizer, epoch, writer)
    WriterSingleton.global_step = global_step
    test(model, device, test_loader, global_step, writer)

  if args.model_file is not None:
    print('Saving trained model to file: ', args.model_file)
    model.save(args.model_file)


if __name__ == '__main__':
    main()
