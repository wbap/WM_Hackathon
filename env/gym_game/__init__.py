from gym.envs.registration import register

register(
    id='counting-v0',
    entry_point='gym_game.envs:CountingEnv'
)

register(
    id='dm2s-v0',
    entry_point='gym_game.envs:Dm2sEnv',
    kwargs={'config_file': 'dm2s.par'}
)
