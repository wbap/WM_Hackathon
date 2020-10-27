import numpy as np
from gym_game.envs.active_vision_env import GridUtil

screen_height = 400
screen_width = 800

grid_util = GridUtil(grid_length_x=3, grid_length_y=4, screen_height=screen_height, screen_width=screen_width)

"""
  Range x: 
  0:[0-266.67]   1:[266.67-533.33]  2:[533.33-800]

  Range y:
  0:[0-100]      1:[100-200]        2:[200-300]     3:[300-400]

  0:(133,50)    1:(400,50)       2:666, 50
  3:(133,150)   4:(400,150)      5:666, 150
  6:(133,250)   7:(400,250)      8:666, 250
  9:(133,350)   10:(400,350)     11:666, 350
"""

num_cells = grid_util.num_cells()
print("num_cells = ", num_cells)

print("size of cells (x, y) = {}".format(grid_util.grid_cell_size))

print("\naction --> xy coord")
for action in range(0, num_cells):
  xy = grid_util.action_2_xy(action)
  print("{} --> {}".format(action, xy))

# walk top left to bottom right
print("\nxy --> xy_array")
for xy in range(0, min(screen_height, screen_width), min(screen_height, screen_width)//10):
  xy_array = np.array([xy, xy])
  action = grid_util.xy_2_action(xy_array)
  print("({}, {}) --> {}".format(xy, xy, action))
