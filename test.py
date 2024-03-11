import os

import cv2


def draw_rectangle_and_save(image_path, output_folder, top_left, bottom_right, color=(0, 255, 0), thickness=2):
    # Check if the output folder exists, create if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the image
    image = cv2.imread(image_path)

    # Draw the rectangle on the image
    cv2.rectangle(image, top_left, bottom_right, color, thickness)

    # Determine the output file path
    output_file_path = os.path.join(output_folder, os.path.basename(image_path))

    # Save the image to the output folder
    cv2.imwrite(output_file_path, image)
    print(f"Image saved to {output_file_path}")


# Example usage
image_path = 'exp_3_distance/env2/blur/3/20240306_163835.jpg'  # Update this path to your image file
output_folder = 'temp'  # Update this path to your desired output folder
top_left = (2215, 1400)
bottom_right = (2275, 1480)

# Call the function with the example parameters
draw_rectangle_and_save(image_path, output_folder, top_left, bottom_right)
