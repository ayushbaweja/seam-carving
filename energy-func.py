# Finding the energy of each pixel in the image using gradient magnitude.

from PIL import Image
import numpy as np

try:
    im = Image.open("creation_of_adam.jpeg").convert('L') # converts to greyscale
    im2arr = np.array(im)
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

sobel_filtered = np.zeros_like(im2arr)

for i in range(im2arr.shape[0] - 2):
    for j in range(im2arr.shape[1] - 2):
        gx = np.sum(np.multiply(Gx, im2arr[i:i+3, j:j+3])) # 3x3 kernel in x direction
        gy = np.sum(np.multiply(Gy, im2arr[i:i+3, j:j+3])) # 3x3 kernel in y direction
        sobel_filtered[i+1, j+1] = np.sqrt(gx ** 2 + gy ** 2) # centered at i+1, j+1 pixel

# arr2im = Image.fromarray(im2arr)
# arr2im.show()
sobel_Im = Image.fromarray(sobel_filtered)
sobel_Im.show()