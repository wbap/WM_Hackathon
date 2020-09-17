from gym.envs.registration import register

register(
  id='simple-v0',
  entry_point='gym_game.envs:SimpleEnv',
  kwargs={'config_file': 'env_config.json'}
)

register(
  id='dm2s-v0',
  entry_point='gym_game.envs:Dm2sEnv',
  kwargs={'config_file': 'dm2s.par'}
)

register(
  id='m2s-v0',
  entry_point='gym_game.envs:M2sEnv',
  kwargs={'config_file': 'm2s.par'}
)
