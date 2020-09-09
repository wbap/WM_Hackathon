import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import pygame as pygame
from .pygame_env import PyGameEnv

class SimpleEnv(PyGameEnv):

  GRID_SIZE = 3
  PX_SIZE = 100
  TIMEOUT = 2000

  ACTION_NONE = 0
  ACTION_N = 1
  ACTION_E = 2 
  ACTION_S = 3
  ACTION_W = 4
  NUM_ACTIONS = 5
  NUM_GOALS = 5

  def __init__(self, config_file=None):
    w = self.GRID_SIZE * self.PX_SIZE
    h = self.GRID_SIZE * self.PX_SIZE
    a = self.NUM_ACTIONS
    super().__init__(a, w, h)

  def reset(self):
    super().reset()
    #self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
    self.pose = [0,0]
    self.goal = self.random_goal()
    self.count = 0
    self.goal_time = self.get_time()

  def random_goal(self):
    x = self.np_random.randint(self.GRID_SIZE)
    y = self.np_random.randint(self.GRID_SIZE)
    goal = [x, y]
    return goal

  def update_pose(self, action):
    if action == self.ACTION_N:
      self.pose[1] -= 1
      if self.pose[1] < 0:
        self.pose[1] = self.GRID_SIZE -1
    elif action == self.ACTION_S:
      self.pose[1] += 1
      if self.pose[1] == self.GRID_SIZE:
        self.pose[1] = 0
    elif action == self.ACTION_E:
      self.pose[0] += 1
      if self.pose[0] == self.GRID_SIZE:
        self.pose[0] = 0
    elif action == self.ACTION_W:
      self.pose[0] -= 1
      if self.pose[0] < 0:
        self.pose[0] = self.GRID_SIZE -1

  def at_goal(self):
    if (self.pose[0] == self.goal[0]) and (self.pose[1] == self.goal[1]):
      return True
    return False

  def get_goal_elapsed_time(self):
    time = self.get_time()
    elapsed_time = time - self.goal_time
    return elapsed_time

  def _do_step(self, action, time):
    #return ob, reward, is_complete, info
    self.update_pose(action)
    elapsed_time = self.get_goal_elapsed_time()
    reward = 0.0
    if self.at_goal():
      reward = 1.0
      self.count += 1
      self.goal = self.random_goal()
      self.goal_time = self.get_time()
    elif elapsed_time > self.TIMEOUT:
      self.count += 1
      self.goal = self.random_goal()
      self.goal_time = self.get_time()


    done = False
    if self.count >= self.NUM_GOALS:
      done = True

    observation = self.get_observation()
    additional = {
      'pose': self.pose,
      'goal': self.goal,
      'reward': reward, 
      'action': action,
      'done': done }
    return [observation, reward, done, additional]
   
  def render_screen(self, screen):
    screen.fill((0,0,0))
    YELLOW = (255,255,0)
    BLUE = (0,0,255)

    rect = pygame.Rect(self.goal[0]*self.PX_SIZE, self.goal[1]*self.PX_SIZE, self.PX_SIZE, self.PX_SIZE)
    pygame.draw.rect(screen, BLUE, rect)
    rect = pygame.Rect(self.pose[0]*self.PX_SIZE, self.pose[1]*self.PX_SIZE, self.PX_SIZE, self.PX_SIZE)
    pygame.draw.rect(screen, YELLOW, rect)
