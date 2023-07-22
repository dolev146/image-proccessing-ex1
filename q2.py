import cv2
import numpy as np
import os

def floyd_steinberg_dithering(input_file):
    # Generate output file name
    base_name = os.path.basename(input_file)
    name, ext = os.path.splitext(base_name)
    output_file = f"{name}_dithered{ext}"

    # Load image
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

    # Perform edge detection
    edges = cv2.Canny(img, 100, 200)

    # Convert image to float64 and normalize to [0, 1]
    img = img.astype(np.float64) / 255.0

    # Pad the input image and the edges with zeros (to avoid out of bounds errors)
    padded_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    padded_edges = cv2.copyMakeBorder(edges, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    # Define the error diffusion matrices
    normal_diffusion = np.array([[0, 0, 7/16], [5/16, 3/16, 1/16]])
    edge_diffusion = np.array([[0, 0, 1/16], [1/16, 1/16, 1/16]])

    # Apply Floyd-Steinberg dithering
    for y in range(1, padded_img.shape[0] - 1):
        for x in range(1, padded_img.shape[1] - 1):
            old_pixel = padded_img[y, x]
            new_pixel = round(old_pixel)
            padded_img[y, x] = new_pixel
            error = old_pixel - new_pixel
            if padded_edges[y, x] == 0:
                diffusion_matrix = normal_diffusion
            else:
                diffusion_matrix = edge_diffusion
            padded_img[y:y+2, x-1:x+2] += error * diffusion_matrix
    dithered_img = (padded_img[1:-1, 1:-1] * 255).astype(np.uint8)  # Convert back to uint8 and scale up by 255

    # Save the dithered image
    cv2.imwrite(output_file, dithered_img)

    return output_file


def standard_floyd_steinberg_dithering(input_file):
    # Generate output file name
    base_name = os.path.basename(input_file)
    name, ext = os.path.splitext(base_name)
    output_file = f"{name}_standard_dithered{ext}"

    # Load image
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

    # Convert image to float64 and normalize to [0, 1]
    img = img.astype(np.float64) / 255.0

    # Pad the input image with zeros (to avoid out of bounds errors)
    padded_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    # Define the error diffusion matrix
    diffusion_matrix = np.array([[0, 0, 7/16], [3/16, 5/16, 1/16]])
    diffusion_matrix = diffusion_matrix * 1.5  # Normalize the matrix

    # Apply Floyd-Steinberg dithering
    for y in range(1, padded_img.shape[0] - 1):
        for x in range(1, padded_img.shape[1] - 1):
            old_pixel = padded_img[y, x]
            new_pixel = round(old_pixel)
            padded_img[y, x] = new_pixel
            error = old_pixel - new_pixel
            padded_img[y:y+2, x-1:x+2] += error * diffusion_matrix
    dithered_img = (padded_img[1:-1, 1:-1] * 255).astype(np.uint8)  # Convert back to uint8 and scale up by 255

    # Save the dithered image
    cv2.imwrite(output_file, dithered_img)

    return output_file



modified_output_file = floyd_steinberg_dithering('abc.jpg')
print(f"Saved dithered image with edge-preserving smoothing as {modified_output_file}")

standard_output_file = standard_floyd_steinberg_dithering('abc.jpg')
print(f"Saved dithered image with standard Floyd-Steinberg dithering as {standard_output_file}")
