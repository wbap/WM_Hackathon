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

from cls_module.components.sparse_autoencoder import SparseAutoencoder
from cls_module.components.simple_autoencoder import SimpleAutoencoder

from agent.stub_agent import StubAgent
from gym_game.envs.pygame_dataset import PyGameDataset
from gym_game.stubs.posterior_cortex import PosteriorCortex
from gym_game.envs.stub_agent_env import StubAgentEnv

def train(args, model, device, train_loader, global_step, optimizer, epoch, writer):
  """Trains the model for one epoch."""
  model.train()
  #for batch_idx, (data, target) in enumerate(train_loader):
  for batch_idx, (data) in enumerate(train_loader):
    #data, target = data.to(device), target.to(device)
    #print('Data:', data)
    #print('Batch:', batch_idx)
    data = data.to(device)

    optimizer.zero_grad()
    encoding, output, target = model(data)
    #print('input min/max=', data.min(), data.max())
    #print('encoding shape', encoding.shape)
    #print('DEcoding shape', output.shape)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()

    writer.add_image('train/inputs', torchvision.utils.make_grid(data), global_step)
    # TODO fix this when input has 6 channels...
    #writer.add_image('train/outputs', torchvision.utils.make_grid(output), global_step)
    writer.add_scalar('train/loss', loss, global_step)

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
      _, output, target = model(data)

      writer.add_image('test/inputs', torchvision.utils.make_grid(data), global_step)
      # TODO fix this when input has 6 channels...
      #writer.add_image('test/outputs', torchvision.utils.make_grid(output), global_step)

      test_loss += F.mse_loss(output, data, reduction='sum').item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)

    writer.add_scalar('test/avg_loss', test_loss, global_step)

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
  parser.add_argument('--config', type=str, default='test_configs/sae.json', metavar='N',
                      help='Model configuration (default: test_configs/sae.json')
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

  with open(args.config) as config_file:
    config = json.load(config_file)

  kwargs = {'batch_size': config['batch_size']}

  if use_cuda:
    kwargs.update({
        'num_workers': 1,
        'pin_memory': True,
        'shuffle': True
    })

  writer = SummaryWriter()

  transform = transforms.Compose([
      transforms.ToTensor()
  ])

  env_name = args.env
  print('Making Gym[PyGame] environment:', env_name)
  env_config_file = args.env_config
  print('Env config file:', env_config_file)
  env = gym.make(env_name, config_file=env_config_file)
  print('Env constructed')

  print('Obs. key:', args.env_obs_key)
  dataset = PyGameDataset(key=args.env_obs_key)
  print('Loading pre-generated data from: ', args.env_data_dir) 
  read_ok = dataset.read(args.env_data_dir)
  print('Loaded pre-generated data?', str(read_ok)) 

  env.reset()
  data_shape = dataset.get_shape(env)
  print('Data shape:', data_shape)

  # train_dataset = datasets.MNIST('./data', train=True, download=True,
  #                                transform=transform)
  # test_dataset = datasets.MNIST('./data', train=False,
  #                               transform=transform)

  train_loader = torch.utils.data.DataLoader(dataset, **kwargs)
  test_loader = torch.utils.data.DataLoader(dataset, **kwargs)

  input_shape = (-1,) + data_shape #[-1, 1, 28, 28]
  print('Final dataset shape:', input_shape)

  # if config['model'] == 'sae':
  #   model = SparseAutoencoder(input_shape, config['model_config']).to(device)
  # elif config['model'] == 'ae':
  #   model = SimpleAutoencoder(input_shape, config['model_config']).to(device)
  # else:
  #   raise NotImplementedError('Model not supported: ' + str(config['model']))
  # model.float()
  
  #env_observation_space = self.env.observation_space
  #obs_keys = [args.env_obs_key]  # pretrain just one
  obs_key = args.env_obs_key
  cortex_config = PosteriorCortex.get_default_config()
  model = PosteriorCortex(obs_key, input_shape, cortex_config)
  print('Model:', model)

  optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

  global_step = 0
  for epoch in range(0, args.epochs):
    global_step = train(args, model, device, train_loader, global_step, optimizer, epoch, writer)
    test(model, device, test_loader, global_step, writer)

  if args.model_file is not None:
    print('Saving trained model to file: ', args.model_file)
    torch.save(model.state_dict(), args.model_file)


if __name__ == '__main__':
    main()
