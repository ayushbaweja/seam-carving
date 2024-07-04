# Finding the energy of each pixel in the image using gradient magnitude.

from PIL import Image
import numpy as np

try:
    im = Image.open("creation_of_adam.jpeg").convert('L') # converts to greyscale
except Exception as e:
    print(e)

# print(im.format, im.size, im.mode)
# im.show()

# print("Width:", im.width)
# print("Height:", im.height)

"""
Sobel operator

      _               _                   _                _
     |                 |                 |                  |
     | 1.0   0.0  -1.0 |                 |  1.0   2.0   1.0 |
Gx = | 2.0   0.0  -2.0 |    and     Gy = |  0.0   0.0   0.0 |
     | 1.0   0.0  -1.0 |                 | -1.0  -2.0  -1.0 |
     |_               _|                 |_                _|

"""

Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, -0.0], [-1.0, -2.0, -1.0]])

sobel_filtered = np.zeros(shape=im.size)

for i in range(im.width - 2):
    for j in range(im.height - 2):
        # do some matrix multiplcation
        break