import os
from PIL import Image
import cv2


def crop_square(image_path, output_path, top_left, bottom_right):
    """
    Crop a square from an image and save the cropped part based on top-left and bottom-right coordinates.
    """
    # Unpack the coordinates
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Open the image
    with Image.open(image_path) as img:
        # Define the bounding box with the given coordinates
        box = (x1, y1, x2, y2)

        # Crop the image
        cropped_image = img.crop(box)

        # Save the cropped image
        cropped_image.save(output_path)


def crop_images_in_folder(source_folder, target_folder, top_left, bottom_right):
    """
    Crop all images in a source folder and save them in a target folder with incremental filenames.
    """
    # Create target folder if it does not exist
    os.makedirs(target_folder, exist_ok=True)

    # List all files in the source folder
    files = os.listdir(source_folder)

    # Loop through each file and cropped_man it
    for i, filename in enumerate(files, start=1):
        # Construct full file path
        file_path = os.path.join(source_folder, filename)
        # Construct output file path
        output_path = os.path.join(target_folder, f"{i}.jpg")
        # Crop and save the image
        crop_square(file_path, output_path, top_left, bottom_right)


# Specify source and target folders
source_folder = 'exp_3_distance/env2/blur/3'
target_folder = 'exp_3_distance/cropped/3.3'
top_left = (2215, 1400)
bottom_right = (2275, 1480)
# Crop all images in the folder
crop_images_in_folder(source_folder, target_folder, top_left, bottom_right)
