

import os
import pygame as pygame
from pygame.locals import *
import logging
from .finite_state_env import FiniteStateEnv


class Dm2sEnv(FiniteStateEnv):

  # Define states of the game
  STATE_TUTOR_STIM = 'tutor-stim'
  STATE_TUTOR_HIDE = 'tutor-hide'
  STATE_TUTOR_SHOW = 'tutor-show'
  STATE_TUTOR_FEEDBACK = 'tutor-feedback'
  STATE_INTER = 'inter'
  STATE_PLAY_STIM = 'play-stim'
  STATE_PLAY_HIDE = 'play-hide'
  STATE_PLAY_SHOW = 'play-show'
  STATE_PLAY_FEEDBACK = 'play-feedback'
  STATE_END = 'end'

  # Define actions
  ACTION_NONE = 0
  ACTION_LEFT = 1
  ACTION_RIGHT = 2
  NUM_ACTIONS = 3  # No action, Left and Right

  POSITION_NONE = 0
  POSITION_L = 1
  POSITION_R = 2

  RESULT_WRONG = 0
  RESULT_CORRECT = 1

  # define colors
  BLACK = (0, 0, 0)
  WHITE = (255, 255, 255)
  YELLOW = (255, 255, 0)
  BLUE = (0, 0, 255)

  # define global variables
  gParams = {}  # parameter dictionary
  gVideoWidth = 800
  gVideoHeight = 800
  gColors = ["LB"]
  gShapes = ["Barred_Ring", "Triangle", "Crescent", "Cross", "Circle", "Heart", "Pentagon", "Ring", "Square"]
  gCorrectFB = Rect(0, gVideoHeight - 80, gVideoWidth, 80)
  gObservBar = Rect(0, 0, gVideoWidth, 80)

  def __init__(self, config_file=None):
    # obtain parameters from a file
    print('D2MS Config file:', config_file)
    with open(config_file) as f:
      for line in f:
        buf = line.strip().split(",")
        self.gParams[buf[0]] = buf[1]
    print('D2MS gParams:', self.gParams)
    #self.image_dir = 'png'
    self.inter_flash = 400
    self.feedback_flash = 100
    self.image_dir = str(self.gParams["imageDir"])
    
    # load button images
    self.gButton1 = self.read_image(self.get_image_file_name('Button_Green'))
    self.gButton1F = self.read_image(self.get_image_file_name('Button_LG'))
    self.gButton2 = self.read_image(self.get_image_file_name('Button_Red'))
    self.gButton2F = self.read_image(self.get_image_file_name('Button_Pink'))

    self.tutor_repeats = int(self.gParams["observationRepeat"])
    self.play_repeats = int(self.gParams["mainTaskRepeat"])
    w = self.gVideoWidth
    h = self.gVideoHeight

    self.sample = None
    self.target = None
    self.position = None
    self.result = None
    self.tutor_counts = 0
    self.play_counts = 0

    # gaze
    self._mode_no_tutor_long_game = False   # currently being used for testing gaze
    self._draw_gaze = True

    # Create base class stuff
    frame_rate = int(self.gParams["frameRate"])
    super().__init__(self.NUM_ACTIONS, w, h, frame_rate)

  def _create_states(self):
    show_stim_interval = 2000 if not self._mode_no_tutor_long_game else 500
    hide_stim_interval = 2000 if not self._mode_no_tutor_long_game else 500
    tutor_show_interval = 1000
    play_show_interval = 5000 if not self._mode_no_tutor_long_game else 20000
    feedback_interval = 1000 if not self._mode_no_tutor_long_game else 500
    inter_interval = 400 * 3

    # if in mode 'no tutor, long game', don't show tutor
    if not self._mode_no_tutor_long_game:
      self.add_state(self.STATE_TUTOR_STIM, next_states=[self.STATE_TUTOR_HIDE], duration=show_stim_interval,
                     start_state=True)
      self.add_state(self.STATE_TUTOR_HIDE, next_states=[self.STATE_TUTOR_SHOW], duration=hide_stim_interval)
      self.add_state(self.STATE_TUTOR_SHOW, next_states=[self.STATE_TUTOR_FEEDBACK], duration=tutor_show_interval)
      self.add_state(self.STATE_TUTOR_FEEDBACK, next_states=[self.STATE_INTER], duration=feedback_interval)

      self.add_state(self.STATE_INTER, next_states=[self.STATE_PLAY_STIM], duration=inter_interval)

    self.add_state(self.STATE_PLAY_STIM, next_states=[self.STATE_PLAY_HIDE], duration=show_stim_interval,
                   start_state=self._mode_no_tutor_long_game)
    self.add_state(self.STATE_PLAY_HIDE, next_states=[self.STATE_PLAY_SHOW], duration=hide_stim_interval)
    self.add_state(self.STATE_PLAY_SHOW, next_states=[self.STATE_PLAY_FEEDBACK], duration=play_show_interval)
    self.add_state(self.STATE_PLAY_FEEDBACK, next_states=[self.STATE_END], duration=feedback_interval)

    self.add_state(self.STATE_END, end_state=True)

  def _get_caption(self):
    return 'Delayed Match-to-Sample'

  def reset(self):
    self.sample = None
    self.target = None
    self.position = None
    self.result = None
    self.tutor_counts = 0
    self.play_counts = 0
    return super().reset()

  def on_state_changed(self, old_state_key, new_state_key):
    logging.info('State -> ', new_state_key, '@t=', self.state_time)
    if new_state_key == self.STATE_TUTOR_STIM or new_state_key == self.STATE_PLAY_STIM:
      self.position = self.np_random.randint(2)+1  # Left:1 & Right:2
      self.sample = self.get_random_sample() 
      self.target = self.get_random_sample(self.sample) 
      self.result = None
      
  def _update_state_key(self, old_state_key, action, elapsed_time):

    # Don't transition from end states
    is_end_state = self.is_end_state(old_state_key)
    if is_end_state:
      return old_state_key  # terminal state

    # transition on response, when appropriate
    new_state_key = old_state_key  # default
    next_state_keys = self.get_next_state_keys(old_state_key)

    # Only transition if action in scope of dm2s actions
    if action < self.NUM_ACTIONS:
      if old_state_key == self.STATE_PLAY_SHOW:
        if action != self.ACTION_NONE:
          if self.result is None:
            if action == self.position:
              self.result = self.RESULT_CORRECT
            else:
              self.result = self.RESULT_WRONG
            print("Response: "+str(action)+", Correct response: "+str(self.position))
          new_state_key = next_state_keys[0]
          return new_state_key
        # else: wait for timer

    # All other states - check timer
    state = self.states[old_state_key]
    duration = state['duration']
    if duration is not None:
      if elapsed_time > duration:
        new_state_key = next_state_keys[0]

        if old_state_key == self.STATE_PLAY_SHOW:
          print("Response: None, Correct response: "+str(self.position))

        # Repeat certain sections again and again        
        self.tutor_repeats = int(self.gParams["observationRepeat"])
        self.play_repeats = int(self.gParams["mainTaskRepeat"])
        if old_state_key == self.STATE_TUTOR_FEEDBACK:
          self.tutor_counts += 1
          if self.tutor_counts < self.tutor_repeats:
            new_state_key = self.STATE_TUTOR_STIM
        elif old_state_key == self.STATE_PLAY_FEEDBACK:
          self.play_counts += 1
          if self.play_counts < self.play_repeats:
            new_state_key = self.STATE_PLAY_STIM
        return new_state_key
    return old_state_key

  def _update_reward(self, old_state_key, action, elapsed_time, new_state_key):
    reward = 0.0
    if old_state_key == self.STATE_PLAY_SHOW:
      if action == self.position:
        reward = 1.0
    return reward

  def get_random_sample(self, unlike_sample=None):
    if unlike_sample is not None:
      except_color = unlike_sample['color']
      except_shape = unlike_sample['shape']
    while(True):
      color = self.gColors[self.np_random.randint(0, len(self.gColors))-1]
      shape = self.gShapes[self.np_random.randint(0, len(self.gShapes))-1]
      if unlike_sample is None:
        break
      elif except_color != color or except_shape != shape:
        break
    sample = {
      'color': color,
      'shape': shape
    }
    return sample

  def get_image_file_name(self, prefix):
    file_name = prefix + '.png'
    return file_name

  def get_sample_file_name(self, sample):
    color = sample['color']
    shape = sample['shape']
    prefix = shape+'_'+color
    return self.get_image_file_name(prefix)

  def read_image(self, file_name):
    file_path = os.path.join(self.image_dir, file_name)
    image = pygame.image.load(file_path)
    return image

  def read_sample_image(self, sample):
    name = self.get_sample_file_name(sample)
    image = self.read_image(name)
    return image

  def render_screen(self, screen):
    state = self.get_state_key()
    if state == self.STATE_END:
      screen.fill(self.BLACK)
      return

    elapsed_time = self.get_state_elapsed_time()
    if state == self.STATE_INTER:
      # flash whole screen
      elapsed_coarse = int(elapsed_time / self.inter_flash)
      if (elapsed_coarse % 2) == 1:
        screen.fill(self.WHITE)
      else:
        screen.fill(self.BLACK)
      return

    # fill screen
    screen.fill(self.WHITE)

    show_tutoring = False
    if state == self.STATE_TUTOR_STIM or state == self.STATE_TUTOR_HIDE or state == self.STATE_TUTOR_SHOW or state == self.STATE_TUTOR_FEEDBACK:
      show_tutoring = True 

    show_correct = False
    if state == self.STATE_PLAY_FEEDBACK and self.result == self.RESULT_CORRECT:
      show_correct = True

    flash = None
    if state == self.STATE_TUTOR_FEEDBACK or state == self.STATE_PLAY_FEEDBACK:
      # flash the correct button
      elapsed_coarse = int(elapsed_time / self.feedback_flash)
      if (elapsed_coarse % 2) == 1:
        flash = self.position

    show_sample = False
    if state == self.STATE_TUTOR_STIM or state == self.STATE_PLAY_STIM:
      show_sample = True

    show_targets = False
    if state == self.STATE_TUTOR_SHOW or state == self.STATE_PLAY_SHOW:
      show_targets = True

    self.draw_screen(screen, tutoring=show_tutoring, correct=show_correct, sample=show_sample, targets=show_targets, flash=flash)

  def draw_screen(self, screen, tutoring=False, correct=False, sample=False, targets=False, flash=None):
    # fill screen
    screen.fill(self.WHITE)

    if correct:
        # draw Result Bar 
        pygame.draw.rect(screen, self.YELLOW, self.gCorrectFB)

    if tutoring:
        # draw Observation Bar (ie being tutored)
        pygame.draw.rect(screen, self.BLUE, self.gObservBar)

    if sample:
      sample_image = self.read_sample_image(self.sample)
      screen.blit(sample_image, (int(self.gVideoWidth/2 - sample_image.get_width()/2), 140))

    if targets:
      if self.position == self.POSITION_L:
        target1_image = self.read_sample_image(self.sample)
        target2_image = self.read_sample_image(self.target)
      else:
        target1_image = self.read_sample_image(self.target)        
        target2_image = self.read_sample_image(self.sample)
      screen.blit(target1_image, (int(self.gVideoWidth/2 - target1_image.get_width()/2) - 160, 410))
      screen.blit(target2_image, (int(self.gVideoWidth/2 - target2_image.get_width()/2) + 160, 410))

    # draw button images
    gButton1 = self.gButton1  # default
    gButton2 = self.gButton2  # default
    if flash is not None:
      if flash == self.POSITION_L:
        gButton1 = self.gButton1F
      elif flash == self.POSITION_R:
        gButton2 = self.gButton2F

    screen.blit(gButton1, (int(self.gVideoWidth/2 - self.gButton1.get_width()/2) - 160, 610))
    screen.blit(gButton2, (int(self.gVideoWidth/2 - self.gButton2.get_width()/2) + 160, 610))

    # draw the gaze position
    if self._draw_gaze:
      pygame.draw.circle(screen, self.BLACK, self.gaze, self.fov_size[0], 1)
      # vertical line
      pygame.draw.line(screen, self.BLACK, self.gaze+[0, self.fov_size[1]], self.gaze+[0, -self.fov_size[1]])
      # horizontal line
      pygame.draw.line(screen, self.BLACK, self.gaze+[self.fov_size[0], 0], self.gaze+[-self.fov_size[0], 0])
