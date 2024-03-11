import cv2
import os
import glob


def gaussian_blur_images(input_folder, output_folder, blur_value=(21, 21)):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all image files in the input folder
    image_files = glob.glob(os.path.join(input_folder, '*'))
    image_count = 0

    for image_path in image_files:
        # Read the image
        image = cv2.imread(image_path)

        # Check if image is loaded successfully
        if image is not None:
            blurred_image = cv2.GaussianBlur(image, blur_value, 0)

            # Save the blurred image to the output folder
            cv2.imwrite(os.path.join(output_folder, f'{image_count + 1}.jpg'), blurred_image)
            print(f'Blurred image saved as {image_count + 1}.jpg')

            image_count += 1
        else:
            print(f'Failed to load image: {image_path}')


if __name__ == '__main__':
    input_folder_path = 'exp_2_lens/blur/7'
    output_folder_path = 'exp_2_lens/blur/8'
    gaussian_blur_images(input_folder_path, output_folder_path)
