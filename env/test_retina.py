import sys
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image

from gym_game.stubs.retina import Retina

config = {
  'f_size': 7,
  'f_sigma': 2.0,
  'f_k': 1.6  # approximates Laplacian of Gaussian
}

channels = 3
model = Retina(channels=channels, config=config)

file_path = sys.argv[1]
print('Loading image: ', file_path)
img = Image.open(file_path)
if channels == 1:
  img = img.convert("L")
print(img.size)

img_tensor = torchvision.transforms.ToTensor()(img)
img_tensor_shape = img_tensor.shape
img_tensor = torch.unsqueeze(img_tensor, 0)  # insert batch dimensions

dog, dog_pos_tensor, dog_neg_tensor = model(img_tensor)

print('dog shape', dog_pos_tensor.shape)
# remove batch and channel dimensions
dog_pos_tensor = torch.squeeze(dog_pos_tensor)
dog_neg_tensor = torch.squeeze(dog_neg_tensor)

dog_pos = torchvision.transforms.ToPILImage()(dog_pos_tensor)
dog_neg = torchvision.transforms.ToPILImage()(dog_neg_tensor)
print('dog shape squeezed', dog_pos_tensor.shape)
print(dog_pos.size)
print(dog_neg.size)

# show with PIL
img.show()
dog_pos.show()
dog_neg.show()

# show in matplotlib figure   --> currently has weird colours, maybe b/c range [0, 256] instead of [-1,1]
fig = plt.figure()

ax = fig.add_subplot(1, 3, 1)
imgplot = plt.imshow(img)
ax.set_title('Original')

ax = fig.add_subplot(1, 3, 2)
imgplot = plt.imshow(dog_pos)
ax.set_title('DoG+')

ax = fig.add_subplot(1, 3, 3)
imgplot = plt.imshow(dog_neg)
ax.set_title('DoG-')

plt.show()
plt.savefig('results.png')
