#!/usr/bin/env python
import sys, gym, time

#
# Test yourself as a learning agent! Pass environment name as a command-line argument, for example:
#
# python keyboard_agent.py SpaceInvadersNoFrameskip-v4
#
import gym_game
import pygame

if len(sys.argv) < 3:
    print('Usage: python keyboard_agent.py ENV CONFIG_FILE')
    sys.exit(-1)

env_name = sys.argv[1]
print('Making Gym[PyGame] environment:', env_name)
config_file = sys.argv[2]
print('Config file:', config_file)
env = gym.make(env_name, config_file=config_file)

sleep_time = 0.1
if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n

print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ... etc.")
print("No keys pressed is taking action 0")

render_mode = 'human'
#render_mode = 'rgb_array'
env.render(render_mode)

def get_action(pressed_keys):
    action = None
    if pressed_keys[pygame.K_0] == 1:
        action = 0
    elif pressed_keys[pygame.K_1] == 1:
        action = 1
    elif pressed_keys[pygame.K_2] == 1:
        action = 2
    elif pressed_keys[pygame.K_3] == 1:
        action = 3
    elif pressed_keys[pygame.K_4] == 1:
        action = 4
    elif pressed_keys[pygame.K_5] == 1:
        action = 5
    elif pressed_keys[pygame.K_6] == 1:
        action = 6
    elif pressed_keys[pygame.K_7] == 1:
        action = 7
    elif pressed_keys[pygame.K_8] == 1:
        action = 8
    elif pressed_keys[pygame.K_9] == 1:
        action = 9
    if action is None:
        action = 0
    return action

def rollout(env):
    observation = env.reset()
    quit = False
    total_reward = 0
    total_timesteps = 0
    while 1:
        # Check for quit from user
        events = env.get_events()
        for event in events:
            if event.type == pygame.QUIT:
                quit = True
                print('Quit event')

        # Get selected action from user
        pressed_keys = env.get_keys_pressed()
        a = get_action(pressed_keys)

        # Update the environment
        observation, reward, done, info = env.step(a)
        total_timesteps += 1
        total_reward += reward
        #print('Obs: ',str(observation))

        # Render the new state
        img = env.render(mode=render_mode, close=quit)  # Render the game

        # Handle quit request
        if quit:
            print('Quitting (truncating rollout)...')
            break
        if done: 
            print('Episode (rollout) complete.')
            env.reset()
            break

        # Wait a short time
        time.sleep(sleep_time)
    print("Rollout summary: Timesteps %i Reward %0.2f" % (total_timesteps, total_reward))
    return quit

while 1:
    quit = rollout(env)
    if quit: break