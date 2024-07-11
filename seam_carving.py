# Seam Carving
# Content-Aware Image Resizing

from PIL import Image
import numpy as np

# loading image
def load_image(image_path, mode = "L"):
    try:
        im = Image.open(image_path).convert(mode) # converts to greyscale
        im2arr = np.array(im)
        return im2arr
    except Exception as e:
        print(e)
        return None

# print(im.format, im.size, im.mode)
# im.show()

# show image from np array
def show_imarr(imarr):
    im = Image.fromarray(imarr)
    im.show()

# print("Width:", im.width)
# print("Height:", im.height)

# Edge detection using Sobel filter
"""
Sobel filter

      _               _                   _                _
     |                 |                 |                  |
     | 1.0   0.0  -1.0 |                 |  1.0   2.0   1.0 |
Gx = | 2.0   0.0  -2.0 |    and     Gy = |  0.0   0.0   0.0 |
     | 1.0   0.0  -1.0 |                 | -1.0  -2.0  -1.0 |
     |_               _|                 |_                _|

"""

def apply_sobel_filter(im2arr):

    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, -0.0], [-1.0, -2.0, -1.0]])

    sobel_filtered = np.zeros_like(im2arr)

    for i in range(im2arr.shape[0] - 2):
        for j in range(im2arr.shape[1] - 2):
            gx = np.sum(np.multiply(Gx, im2arr[i:i+3, j:j+3])) # 3x3 kernel in x direction
            gy = np.sum(np.multiply(Gy, im2arr[i:i+3, j:j+3])) # 3x3 kernel in y direction
            sobel_filtered[i+1, j+1] = np.sqrt(gx ** 2 + gy ** 2) # centered at i+1, j+1 pixel
    return sobel_filtered

    # show_image(im2arr)
    # show_image(sobel_filtered)

# finding min seam
def min_seam(im2arr):
    energy = apply_sobel_filter(im2arr)
    dp = np.zeros_like(energy)

    for i in range(1, im2arr.shape[0]):
        for j in range(0, im2arr.shape[1]):
            if j==0: # left edge
                idx = np.argmin(energy[i-1, j:j+2])
                dp[i,j] = idx+j
                min_energy = energy[i-1, idx+j]
            else:
                idx = np.argmin(energy[i - 1, j - 1:j + 2])
                dp[i, j] = idx + j - 1
                min_energy = energy[i - 1, idx + j - 1]

            energy[i, j] += min_energy

    return energy, dp

# removing seam
def carve(im2arr):
    energy, dp = min_seam(im2arr)


# running
image_path = "example/surfer.jpg"
im2arr = load_image(image_path)

if im2arr is not None:
    sobel_filtered = apply_sobel_filter(im2arr)
    show_imarr(sobel_filtered)
