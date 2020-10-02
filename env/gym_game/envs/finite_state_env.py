
from abc import ABC, abstractmethod
import logging
from .active_vision_env import ActiveVisionEnv


class FiniteStateEnv(ActiveVisionEnv):

  def __init__(self, num_actions, screen_width, screen_height, frame_rate):

    # Create state machine
    self.states = {}
    self.start_state = None
    self.state_key = None
    self.state_time = None
    self._create_states()

    super().__init__(num_actions, screen_width, screen_height, frame_rate)

  @abstractmethod
  def _create_states(self):
    """Override this method to define the state-graph"""
    pass

  @abstractmethod
  def _update_state_key(self, old_state_key, action, elapsed_time):
    """
    Return the new state key, or the old state key if unchanged.
    :rtype: state_key
    """
    pass

  @abstractmethod
  def _update_reward(self, old_state_key, action, elapsed_time, new_state_key):
    """
    Return the reward for the specified old-state/new-state given action,time.
    :rtype: real
    """
    pass

  def reset(self):
    #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~ RESET ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    assert(self.start_state is not None)
    self.state_key = None
    self.state_time = None
    self.reset_time()
    self.set_state(self.start_state)
    #return self.get_observation()
    obs = super().reset()
    #self.state_time = self.get_time()  # because the clock has been reset
    return obs

  def add_state(self, state_key, start_state=False, end_state=False, next_states=[], duration=None, meta=None):
    state = {
      'key': state_key,
      'start': start_state,
      'end': end_state,
      'duration': duration,
      'next_state_keys': next_states,
      'meta': meta
    }    
    self.states[state_key] = state
    if start_state is True:
      self.start_state = state_key
      logging.debug('Start state = ', state_key)
    logging.debug('Adding state:', str(state))

  def set_state(self, state_key):
    old_state_key = self.state_key
    time = self.get_time()  # Record the time of entering the state
    self.state_key = state_key  # The key, not the object
    self.state_time = time  # When we entered the state
    #print('~~~~~~ SET STATE:', state_key, 'to time:', self.state_time)
    self.on_state_changed(old_state_key, state_key)

  def on_state_changed(self, old_state_key, new_state_key):
    logging.info('State -> ', self.state_key, '@t=', self.state_time)

  def get_state_key(self):
    return self.state_key

  def get_state_elapsed_time(self):
    time = self.get_time()
    elapsed_time = time - self.state_time
    return elapsed_time

  def is_end_state(self, state_key):
    state = self.states[state_key]
    return state['end']

  def get_next_state_keys(self, state_key):
    state = self.states[state_key]
    next_states = state['next_state_keys']
    return next_states

  def _do_step(self, action, time):
    super()._do_step(action, time)
    old_state_key = self.get_state_key()
    elapsed_time = self.get_state_elapsed_time()
    logging.debug('old state=', old_state_key, 'time=', elapsed_time)
    #print('old state=', old_state_key, 'state time=', elapsed_time)
    new_state_key = self._update_state_key(old_state_key, action, elapsed_time)
    reward = self._update_reward(old_state_key, action, elapsed_time, new_state_key)

    if new_state_key != old_state_key:
      self.set_state(new_state_key)

    observation = self.get_observation()
    is_end_state = self.is_end_state(new_state_key)

    self.reward = reward
    additional = {
      'old_state': old_state_key,
      'new_state': new_state_key,
      'reward': self.reward, 
      'action': action,
      'done': is_end_state}
    logging.info('additional:', str(additional))
    return [observation, self.reward, is_end_state, additional]
