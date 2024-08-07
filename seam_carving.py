# Seam Carving
# Content-Aware Image Resizing

from PIL import Image
import numpy as np
import argparse

# loading image
def load_image(image_path, mode = "RGB"):
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

# gradient magnitude

def compute_energy(im2arr):

    # Convert to grayscale for energy calculation
    gray_arr = np.dot(im2arr[...,:3], [0.2989, 0.5870, 0.1140])

    gx = np.zeros_like(gray_arr)
    gy = np.zeros_like(gray_arr)

    gx[:, 1:-1] = gray_arr[:, 2:] - gray_arr[:, :-2] # difference between pixel to the right and pixel to the left
    gx[:, 0] = gray_arr[:, 1] - gray_arr[:, 0] # for first column
    gx[:,-1] = gray_arr[:,-1] - gray_arr[:, -2] # for last column

    gy[1:-1, :] = gray_arr[2:, :] - gray_arr[:-2, :] # difference between pixel below and pixel above
    gy[0, :] = gray_arr[1, :] - gray_arr[0, :] # for first row
    gy[-1, :] = gray_arr[-1, :] - gray_arr[-2, :] # for last row

    return np.sqrt(gx**2 + gy**2)
    # show_image(im2arr)

# seam values
def seam_val(im2arr):
    energy = compute_energy(im2arr)
    dp = np.zeros_like(energy, dtype=np.float64)

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

# removing lowest seam
def carve(im2arr):
    energy, dp = seam_val(im2arr)
    seam = np.zeros(im2arr.shape[0], dtype=int)  # create a 1D array for the seam

    # column index of minimum energy in last row
    seam[-1] = np.argmin(energy[-1])

    # tracing seam
    for i in range(im2arr.shape[0] - 2, -1, -1):
        seam[i] = dp[i+1, seam[i+1]]

    carved_image = np.zeros((im2arr.shape[0], im2arr.shape[1] - 1, 3), dtype=im2arr.dtype)
    for i in range(im2arr.shape[0]):
        carved_image[i, :, :] = np.delete(im2arr[i, :, :], seam[i], axis=0)

    return carved_image


# running
def main():
    parser = argparse.ArgumentParser(description='Seam Carving to reduce image width.')
    parser.add_argument('input_image', type=str, help='Path to the input image')
    parser.add_argument('output_image', type=str, help='Path to save the carved image')
    parser.add_argument('new_width', type=int, help='Desired width of the output image')

    args = parser.parse_args()

    # Load the input image
    im2arr = load_image(args.input_image)
    if im2arr is None:
        print(f"Error: Unable to open image file {args.input_image}")
        return

    # Calculate the number of seams to remove
    num_seams = im2arr.shape[1] - args.new_width
    if num_seams <= 0:
        print(f"Error: The desired width {args.new_width} is not smaller than the current width {im2arr.shape[1]}")
        return

    # Carve the seams iteratively
    for _ in range(num_seams):
        print("carving!")
        im2arr = carve(im2arr)

    # Save the carved image
    carved_image = Image.fromarray(im2arr.astype("uint8"), 'RGB')
    carved_image.save(args.output_image)
    print(f"Carved image saved to {args.output_image}")

if __name__ == "__main__":
    main()
