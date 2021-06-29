
import json
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
  GRAY = (128, 128, 128)
  RED = (255, 0, 0)

  GAME_TYPE_SHAPE = 'shape'
  GAME_TYPE_COLOR = 'color'
  GAME_TYPE_POSITION = 'position'

  TX_TYPES = ["None", "Reduce", "Rotate"]
  BAR_POSITIONS = ['Bottom', 'Right', 'Left', 'Top']

  # define global variables
  gParams = {}  # parameter dictionary

  def __init__(self, config_file=None):
    # obtain parameters from a file
    print('Env config file:', config_file)
    # with open(config_file) as f:
    #   for line in f:
    #     buf = line.strip().split(",")
    #     self.gParams[buf[0]] = buf[1]
    with open(config_file) as json_file:
      self.gParams = json.load(json_file)
    print('D2MS gParams:', self.gParams)
    #self.image_dir = 'png'
    self.gVideoWidth = self.gParams["videoWidth"]
    self.gVideoHeight = self.gParams["videoHeight"]
    self.gameTypes = self.gParams['gameTypes']
    self.txTypes = self.gParams['txTypes']
    self.gShapes = self.gParams["shapes"]
    self.gColors = self.gParams["colors"]
    self.gCorrectFB = Rect(0, self.gVideoHeight - 80, self.gVideoWidth, 80)
    self.gObservBar = Rect(0, 0, self.gVideoWidth, 80)

    #self.inter_flash = 400
    #self.feedback_flash = 100
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
    self.sample_correct = None  # The actual, correct symbol shown as a stimulus
    self.sample_wrong = None  # An incorrect, distractor stimulus symbol
    self.position = None  # Where the actual symbol is shown during the choice
    self.result = None
    self.tutor_counts = 0
    self.play_counts = 0

    # Create base class stuff
    frame_rate = int(self.gParams["frameRate"])
    super().__init__(self.NUM_ACTIONS, w, h, frame_rate)

  def get_config(self):
    return self.gParams

  def _create_states(self):
    def get_interval(state):
      interval = self.gParams['states'][state]['interval']
      return interval

    self.add_state(self.STATE_TUTOR_STIM, next_states=[self.STATE_TUTOR_HIDE], duration=get_interval(self.STATE_TUTOR_STIM), start_state=True)
    self.add_state(self.STATE_TUTOR_HIDE, next_states=[self.STATE_TUTOR_SHOW], duration=get_interval(self.STATE_TUTOR_HIDE))
    self.add_state(self.STATE_TUTOR_SHOW, next_states=[self.STATE_TUTOR_FEEDBACK], duration=get_interval(self.STATE_TUTOR_SHOW))
    self.add_state(self.STATE_TUTOR_FEEDBACK, next_states=[self.STATE_INTER], duration=get_interval(self.STATE_TUTOR_FEEDBACK))

    self.add_state(self.STATE_INTER, next_states=[self.STATE_PLAY_STIM], duration=get_interval(self.STATE_INTER))

    self.add_state(self.STATE_PLAY_STIM, next_states=[self.STATE_PLAY_HIDE], duration=get_interval(self.STATE_PLAY_STIM))
    self.add_state(self.STATE_PLAY_HIDE, next_states=[self.STATE_PLAY_SHOW], duration=get_interval(self.STATE_PLAY_HIDE))
    self.add_state(self.STATE_PLAY_SHOW, next_states=[self.STATE_PLAY_FEEDBACK], duration=get_interval(self.STATE_PLAY_SHOW))
    self.add_state(self.STATE_PLAY_FEEDBACK, next_states=[self.STATE_END], duration=get_interval(self.STATE_PLAY_FEEDBACK))

    self.add_state(self.STATE_END, end_state=True)

  def _get_caption(self):
    return 'Delayed Match-to-Sample'

  def reset(self):
    self.sample = None
    self.sample_correct = None
    self.sample_wrong = None
    self.position = None
    self.result = None
    self.tutor_counts = 0
    self.play_counts = 0
    self.gameType = self.get_random_game_type()
    return super().reset()

  def on_state_changed(self, old_state_key, new_state_key):
    logging.info('State -> ', new_state_key, '@t=', self.state_time)
    #print('------------------------------------------------------------------ State -> ', new_state_key, '@t=', self.state_time)
    if new_state_key == self.STATE_TUTOR_STIM or new_state_key == self.STATE_PLAY_STIM:
      self.get_random_samples()

  def get_random_samples(self):
    """Fetch a new set of samples representing original, correct match and incorrect (distractor)"""
    self.position = self.np_random.randint(2)+1  # Left:1 & Right:2
    self.sample = self.get_random_sample() 
    self.sample_correct = self.get_random_sample(like=self.sample)
    self.sample_wrong = self.get_random_sample(unlike=self.sample) 
    self.txType = self.get_random_tx_type()
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
        if action in [self.ACTION_LEFT, self.ACTION_RIGHT]:
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
    """Reward is always zero unless in the PLAY_SHOW state and the action is not zero.
    In that case it is 1 if correct and -1 otherwise."""
    reward = 0.0
    if old_state_key == self.STATE_PLAY_SHOW:
      if action in [self.ACTION_LEFT, self.ACTION_RIGHT]:
        if action == self.position:
          reward = 1.0
        else:
          reward = -1.0
    return reward

  def get_random_game_type(self):
    """Choose a random game type from the list allowed in this instantiation of the environment"""
    numGameTypes = len(self.gameTypes)
    gameTypeIndex = self.np_random.randint(0, numGameTypes)    
    gameType = self.gameTypes[gameTypeIndex]
    print('GAME TYPE: ', gameType)
    return gameType

  def get_random_tx_type(self):
    """Choose a random transform type from the list allowed in this instantiation of the environment"""
    numTxTypes = len(self.txTypes)
    txTypeIndex = self.np_random.randint(0, numTxTypes)    
    txType = self.txTypes[txTypeIndex]
    # print('TX TYPE: ', txType)
    return txType

  def get_random_sample(self, like=None, unlike=None):
    """The rules of the game dictate what the random samples can be. We must avoid more than one
    matching sample in the quality that determines the correct answer in the current game type.
    The matching sample should ONLY be matching in the aspect that matters for the current game type."""
    except_color = None
    except_shape = None
    except_position = None
    if unlike is not None:
      if self.gameType == self.GAME_TYPE_COLOR:
        except_color = unlike['color']
      if self.gameType == self.GAME_TYPE_SHAPE:
        except_shape = unlike['shape']
      if self.gameType == self.GAME_TYPE_POSITION:
        except_position = unlike['position']

    while True:
      color = self.gColors[self.np_random.randint(0, len(self.gColors))-1]
      shape = self.gShapes[self.np_random.randint(0, len(self.gShapes))-1]
      position = self.BAR_POSITIONS[self.np_random.randint(0, len(self.BAR_POSITIONS))-1]

      # If like is defined, then make sure the necessary attribute is LIKE the supplied sample
      if like is not None:
        if self.gameType == self.GAME_TYPE_COLOR:
          color = like['color']
        if self.gameType == self.GAME_TYPE_SHAPE:
          shape = like['shape']
        if self.gameType == self.GAME_TYPE_POSITION:
          position = like['position']

      if unlike is None:
        break  # No constraint
      # Else: Something to avoid      
      if except_color is not None and except_color != color:
        # Don't allow same color, and color is different: OK.
        break
      if except_shape is not None and except_shape != shape:
        # Don't allow same shape, and shape is different: OK.
        break
      if except_position is not None and except_position != position:
        break
    sample = {
      'color': color,
      'shape': shape,
      'position': position
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
      flash_interval = self.get_config()['states'][state]['flash_interval']
      elapsed_coarse = int(elapsed_time / flash_interval)
      if (elapsed_coarse % 2) == 1:
        screen.fill(self.WHITE)
      else:
        screen.fill(self.BLACK)
      return

    # fill screen
    screen.fill(self.WHITE)
    screen_options = self.get_screen_options(state, elapsed_time)
    self.draw_screen(screen, screen_options)

  def get_screen_options(self, state, elapsed_time):
    screen_options = self.get_default_screen_options()

    if state == self.STATE_TUTOR_STIM or state == self.STATE_TUTOR_HIDE or state == self.STATE_TUTOR_SHOW or state == self.STATE_TUTOR_FEEDBACK:
      screen_options['tutoring'] = True

    if state == self.STATE_PLAY_FEEDBACK:
      if self.result == self.RESULT_CORRECT:
        screen_options['correct'] = True
      else:
        screen_options['wrong'] = True        

    if state == self.STATE_TUTOR_FEEDBACK or state == self.STATE_PLAY_FEEDBACK:
      # flash the correct button
      flash_interval = self.get_config()['states'][state]['flash_interval']
      elapsed_coarse = int(elapsed_time / flash_interval)
      #print('Elapsed: ', elapsed_time, 'int ', flash_interval, 'coarse ', elapsed_coarse)
      if (elapsed_coarse % 2) == 1:
        #flash = self.position
        screen_options['flash'] = self.position

    if state == self.STATE_TUTOR_STIM or state == self.STATE_PLAY_STIM:
      screen_options['sample'] = True

    if state == self.STATE_TUTOR_SHOW or state == self.STATE_PLAY_SHOW:
      screen_options['targets'] = True

    return screen_options

  def get_default_screen_options(self):
    screen_options = {
      'tutoring': False,
      'correct': False,
      'wrong': False,
      'sample': False,
      'targets': False,
      'flash': None
    }
    return screen_options

  def draw_sample(self, screen, sample, x, y, txType):
    bar_length = 150  # TODO remove these constants
    bar_height = 20
    bar_dx = 120
    bar_dy = 120
    image = self.read_sample_image(sample)
    image2 = self.transform_image(image, txType)
    halfWidth = int(image2.get_width()/2)
    halfHeight = int(image2.get_height()/2)
    x2 = x - halfWidth  # Center it
    y2 = y - halfHeight
    screen.blit(image2, (x2, y2))

    # Draw bar around the sample? Only in the relevant game.
    if self.gameType == self.GAME_TYPE_POSITION:
      position = sample['position']
      bar_w = bar_length  # Horz bar
      bar_h = bar_height
      if position == 'Left' or position == 'Right':
        bar_w = bar_height
        bar_h = bar_length  # vertical
      bar_x = x - int(bar_w * 0.5)
      bar_y = y - int(bar_h * 0.5)
      # Apply an offset to bar position reflecting the relative position to sample:
      if position == 'Left':
        bar_x -= bar_dx
      elif position == 'Right':
        bar_x += bar_dx
      elif position == 'Top':
        bar_y -= bar_dy
        pass
      elif position == 'Bottom':
        bar_y += bar_dy

      pygame.draw.rect(screen, self.GRAY, (bar_x, bar_y, bar_w, bar_h))

  def transform_image(self, image, txType=None):
    """Transforms an image being used as a sprite."""
    if txType == 'Reduce':
      scale_factor = 0.7
      return pygame.transform.scale(image, (int(image.get_width() * scale_factor), int(image.get_height() * scale_factor)))
    elif txType == 'Rotate': # +45
      angle = 45.0
      return pygame.transform.rotate(image, angle)
    else:
      return image

  def draw_screen(self, screen, screen_options):
    # fill screen
    screen.fill(self.WHITE)

    if screen_options['correct']:
        # draw Result Bar 
        pygame.draw.rect(screen, self.YELLOW, self.gCorrectFB)

    if screen_options['wrong']:
        # draw Result Bar 
        pygame.draw.rect(screen, self.RED, self.gCorrectFB)

    if screen_options['tutoring']:
        # draw Observation Bar (ie being tutored)
        pygame.draw.rect(screen, self.BLUE, self.gObservBar)

    if screen_options['sample']:
      # sample_image = self.read_sample_image(self.sample_correct)
      # screen.blit(sample_image, (int(self.gVideoWidth/2 - sample_image.get_width()/2), 140))
      x = int(self.gVideoWidth/2)
      y = 140
      self.draw_sample(screen, self.sample, x, y, txType='None')

    if screen_options['targets']:
      if self.position == self.POSITION_L:
        # target1_image = self.read_sample_image(self.sample_correct)
        # target2_image = self.read_sample_image(self.sample_wrong)
        option1 = self.sample_correct
        option2 = self.sample_wrong
      else:
        # target1_image = self.read_sample_image(self.sample_wrong)        
        # target2_image = self.read_sample_image(self.sample_correct)
        option1 = self.sample_wrong
        option2 = self.sample_correct
      # screen.blit(target1_image, (int(self.gVideoWidth/2 - target1_image.get_width()/2) - 160, 410))
      # screen.blit(target2_image, (int(self.gVideoWidth/2 - target2_image.get_width()/2) + 160, 410))
      x = int(self.gVideoWidth/2)
      dx = 160  # TODO: Get rid of all these hardcoded constants
      y = 410
      self.draw_sample(screen, option1, x -dx, y, self.txType)
      self.draw_sample(screen, option2, x +dx, y, self.txType)

    # draw button images
    flash = screen_options['flash']
    gButton1 = self.gButton1  # default
    gButton2 = self.gButton2  # default
    if flash is not None:
      if flash == self.POSITION_L:
        gButton1 = self.gButton1F
      elif flash == self.POSITION_R:
        gButton2 = self.gButton2F

    screen.blit(gButton1, (int(self.gVideoWidth/2 - self.gButton1.get_width()/2) - 160, 610))
    screen.blit(gButton2, (int(self.gVideoWidth/2 - self.gButton2.get_width()/2) + 160, 610))

    super().draw_screen(screen, [])
