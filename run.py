import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import mtf as mtf


# Assume other necessary imports are already included

def calculate_and_save_mtf_for_all_images(source_folder, target_folder):
    os.makedirs(target_folder, exist_ok=True)  # Ensure target folder exists

    # List all images in the source folder
    image_files = [f for f in os.listdir(source_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()  # Sort files to ensure order, if necessary

    for idx, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(source_folder, image_file)
        output_path = os.path.join(target_folder, f'{idx}.jpg')

        # Load image as array
        imgArr = mtf.Helper.LoadImageAsArray(image_path)

        # Calculate MTF and save the plot
        mtf_calculated = mtf.MTF.CalculateMtf(imgArr, plot_save_path=output_path, verbose=mtf.Verbosity.DETAIL)


# Example usage
source_folder = 'exp_3_distance/cropped/3.3'
target_folder = 'exp_3_distance/mtf/env_2/3.3'
calculate_and_save_mtf_for_all_images(source_folder, target_folder)
