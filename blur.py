import cv2
import os


def gaussian_blur_images(input_directory, output_directory, blur_intensity=21, passes=3):
    """
    Apply Gaussian blur multiple times to all images in the specified directory and save them with new filenames.

    Parameters:
    - input_directory: Path to the directory containing the original images.
    - output_directory: Path to the directory where blurred images will be saved.
    - blur_intensity: The intensity of the Gaussian blur (kernel size) for each pass. Must be an odd number.
    - passes: The number of times the Gaussian blur is applied.
    """
    # Ensure the blur intensity is odd
    if blur_intensity % 2 == 0:
        raise ValueError("Blur intensity must be an odd number.")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # List all files in the input directory
    files = os.listdir(input_directory)
    image_counter = 1  # Counter for naming the output images

    for file in files:
        # Construct the full file path
        file_path = os.path.join(input_directory, file)

        # Check if the file is an image
        if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Load the image
            image = cv2.imread(file_path)

            # Apply Gaussian blur multiple times
            blurred_image = image
            for _ in range(passes):
                blurred_image = cv2.GaussianBlur(blurred_image, (blur_intensity, blur_intensity), 0)

            # Save the blurred image
            output_file_path = os.path.join(output_directory, f"{image_counter}.jpg")
            cv2.imwrite(output_file_path, blurred_image)

            print(f"Processed {file} -> {output_file_path}")
            image_counter += 1


if __name__ == "__main__":
    input_dir = "exp_1_focus/data_set/blurred/varsha/8"  # Update this path
    output_dir = "exp_1_focus/data_set/blurred/varsha/9"  # Update this path
    gaussian_blur_images(input_dir, output_dir, blur_intensity=21, passes=3)
