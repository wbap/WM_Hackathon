import gym
from gym import error, spaces, utils
from gym.utils import seeding

from abc import ABC, abstractmethod
import numpy as np
import pygame as pygame

from .pygame_env import PyGameEnv

import torch
import torch.utils.data
import torchvision as tv
import torch.utils.data

import bz2
import pickle  # for read/write data

class PyGameDataset(torch.utils.data.Dataset):
  """A PyTorch Dataset made from samples generated from PyGame environment rollouts"""

  def __init__(self):
    pass

  def read(self, file_name):
    """
    Read (Load) from file_name using pickle
    
    @param file_name: name of file to load from
    @type file_name: str
    """
    try:
      f = bz2.BZ2File(file_name, 'rb')
    except (IOError, details):
      sys.stderr.write('File ' + file_name + ' cannot be read\n')
      sys.stderr.write(details)
      return False
    return True

    self._samples = pickle.load(f)
    f.close()

  def write(self, file_name):
    """
    Save Dataset to file using compressed pickle
    
    @param file_name: name of destination file
    @type file_name: str
    """    
    #with open(file_name, 'wb') as f:
    #  # Pickle the 'data' dictionary using the highest protocol available.
    #  pickle.dump(self._samples, f, pickle.HIGHEST_PROTOCOL)
    # file = bz2.BZ2File(file_name, 'w')
    # pickle.dump(self._samples, file)

    try:
      f = bz2.BZ2File(file_name, 'wb')
    except (IOError, details):
      sys.stderr.write('File ' + file_name + ' cannot be written\n')
      sys.stderr.write(details)
      return

    pickle.dump(self._samples, f, pickle.HIGHEST_PROTOCOL)
    f.close()

  def generate(self, num_samples, env, policy, truncate=True):
    """Pregenerate a finite set of samples from the environment"""
    self._samples = []
    while(len(self._samples) < num_samples):
      print('Have ', len(self._samples), ' samples.')
      samples = self.rollout(env, policy)  # perform a rollout
      self._samples = self._samples + samples
    if truncate:
      del self._samples[num_samples:]

  def rollout(self, env, policy):
    """Perform a rollout of the env using the policy"""
    samples = []
    observation = env.reset()
    policy.reset()
    total_reward = 0
    total_timesteps = 0
    while 1:
      samples.append(observation)
      a = policy.get_action(observation)  # Generate policy given observation
      observation, reward, done, info = env.step(a)
      #print('obs shape = ',observation)
      total_timesteps += 1
      total_reward += reward  # for tracking exploration
      if done:
        print('Episode (rollout) complete.')
        env.reset()
        break

    print("Rollout summary: Timesteps %i Reward %0.2f" % (total_timesteps, total_reward))
    return samples

  def __len__(self):
    """Returns the size of the dataset"""
    return len(self._samples)

  def __getitem__(self, idx):
    """Return the ith sample of the dataset"""
    return self._samples[idx]
