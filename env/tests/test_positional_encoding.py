from gym_game.stubs.positional_encoder import PositionalEncoder
import matplotlib.pyplot as plt

pe_config = {'dims': 64}
input_shape = [-1, 2]
pe = PositionalEncoder('pe', input_shape=input_shape, config=pe_config, max_xy=(10, 20))


print(pe.pe_x.shape)
print(pe.pe_y.shape)

fig = plt.figure()

ax = fig.add_subplot(1, 2, 1)
plt.pcolormesh(pe.pe_x, cmap='viridis')
plt.ylim((pe.max_xy[0], 0))
ax.set_title('pe_x')

ax = fig.add_subplot(1, 2, 2)
plt.pcolormesh(pe.pe_y, cmap='viridis')
plt.ylim((pe.max_xy[1], 0))
ax.set_title('pe_y')

plt.colorbar()

plt.show()
plt.savefig('positional_encoding.png')
