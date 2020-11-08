import os
from os import listdir
from os.path import isfile, join

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from abc import ABC, abstractmethod
import numpy as np
import pygame as pygame

from .active_vision_env import WriterSingleton
from .pygame_env import PyGameEnv

import torch
import torch.utils.data
import torchvision as tv
import torch.utils.data

import bz2
import pickle  # for read/write data

class PyGameDataset(torch.utils.data.Dataset):
  """A PyTorch Dataset made from samples generated from PyGame environment rollouts"""

  def __init__(self, key=None):
    self.key = key

  def read(self, dir_name):
    """
    Read (Load) from file_name using pickle
    
    @param file_name: name of file to load from
    @type file_name: str
    """
    # try:
    #   f = bz2.BZ2File(file_name, 'rb')
    # except (IOError, details):
    #   sys.stderr.write('File ' + file_name + ' cannot be read\n')
    #   sys.stderr.write(details)
    #   return False

    # self._samples = pickle.load(f)
    # f.close()
    # return True
    
    # Its important to use binary mode 
    #f = open(file_name, 'ab')   
    self._samples = []
    files = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    num_files = len(files)
    for i in range(0, num_files):
      file_name = files[i]
      file_path = os.path.join(dir_name, file_name)
      print('Reading file:', file_path)
      f = open(file_path, 'rb')   
      sample = pickle.load(f)
      f.close()
      self._samples.append(sample)
    return True

  def write(self, dir_name):
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

    num_samples = len(self._samples)
    for i in range(0, num_samples):
      file_name = str(i) + '.pickle'
      file_path = os.path.join(dir_name, file_name)
      sample = self._samples[i]

      #print('sample:', sample)
      # try:
      #   f = bz2.BZ2File(file_path, 'wb')
      # except (IOError):
      #   sys.stderr.write('File ' + file_path + ' cannot be written\n')
      #   return
      f = open(file_path, 'wb') 

      print('dumping...', i, ' of ', num_samples)
      pickle.dump(sample, f)  #, pickle.HIGHEST_PROTOCOL)
      f.close()

    # # Its important to use binary mode 
    # f = open(file_name, 'ab') 
    # pickle.dump(self._samples, f)
    # f.close()

  def generate(self, num_samples, env, policy, truncate=True):
    """Pregenerate a finite set of samples from the environment"""
    self._samples = []
    WriterSingleton.global_step = 0
    while(len(self._samples) < num_samples):
      print('Have ', len(self._samples), ' samples.')
      samples = self.rollout(env, policy)  # perform a rollout
      self._samples = self._samples + samples
    if truncate:
      del self._samples[num_samples:]

  def get_shape(self, env):
    """Obtains an observation by reset()ing the environment, then returns that shape. Assumes the observation is a numpy array."""
    observation = env.get_observation()
    if self.key is not None:
      return observation[self.key].shape
    return observation.shape

  def rollout(self, env, policy):
    """Perform a rollout of the env using the policy"""
    samples = []
    observation = env.reset()
    #print('obs:', observation)
    policy.reset()
    total_reward = 0
    total_timesteps = 0
    while 1:
      samples.append(observation)
      a = policy.get_action(observation)  # Generate policy given observation
      observation, reward, done, info = env.step(a)
      WriterSingleton.global_step += 1
      #print('obs shape = ',observation)
      total_timesteps += 1
      total_reward += reward  # for tracking exploration
      #print("Rollout summary: Timesteps %i Reward %0.2f" % (total_timesteps, total_reward))
      if done:
        print('Episode (rollout) complete.')
        env.reset()
        break

    print("Rollout summary: Timesteps %i Reward %0.2f" % (total_timesteps, total_reward))
    return samples

  def __len__(self):
    """Returns the size of the dataset"""
    #print('Len?????????????', len(self._samples))
    return len(self._samples)

  def __getitem__(self, idx):
    """Return the ith sample of the dataset"""
    #print('get item?????????????', idx)
    sample = self._samples[idx]
    #print('Sample ', str(sample))
    if self.key is not None:
      sample = sample[self.key]  #.astype(np.float32)
    #print('Sample[key]= ', str(sample))
    #print('Sample dtyp=', sample.dtype)
    return sample
