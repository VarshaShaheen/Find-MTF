import cv2
import numpy as np
import os
import re


def scale_high_frequencies(channel):
    dft = np.fft.fft2(channel)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2

    # Scale frequencies based on their distance from the center
    mask = np.zeros((rows, cols), np.float32)
    for x in range(rows):
        for y in range(cols):
            distance = np.sqrt((x - crow) ** 2 + (y - ccol) ** 2)
            mask[x, y] = np.exp(-4 * distance / (rows / 4))  # Adjusted for more blur

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back


def process_image_color(image):
    # Split the image into its color channels
    channels = cv2.split(image)
    processed_channels = []

    for channel in channels:
        processed_channel = scale_high_frequencies(channel)
        processed_channels.append(processed_channel)

    # Merge the processed channels back into a color image
    processed_image = cv2.merge(processed_channels)
    return processed_image


def extract_number(filename):
    s = re.findall(r"\d+", filename)
    return (int(s[0]) if s else -1, filename)


def process_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    files.sort(key=extract_number)  # Sort files based on numeric order

    for i, file in enumerate(files, start=1):
        file_path = os.path.join(input_folder, file)
        image = cv2.imread(file_path)  # Read in color
        if image is None:
            print(f"Skipping file {file}, as it's not a valid image.")
            continue

        processed_image = process_image_color(image)


# Example usage
input_folder = 'exp_2_lens/blur/2'
output_folder = 'exp_2_lens/blur/3'
process_images_in_folder(input_folder, output_folder)
