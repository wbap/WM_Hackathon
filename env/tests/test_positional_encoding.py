from gym_game.stubs.positional_encoder import PositionalEncoder

pe_config = PositionalEncoder.get_default_config()
input_shape = [-1, 2]
pe = PositionalEncoder('pe', input_shape=input_shape, config=pe_config, max_xy=(400, 800))
