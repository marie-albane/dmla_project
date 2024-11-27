import numpy as np
import pandas as pd
import cv2
import os


def resize_without_distortion(image, target_size):

    """

        This function resizes images without distorting them.

        Parameters:
        image (str): the input image in RGB format (np.array)
        target_size (tuple): desired (width, height)

        Returns:
        list: list of np.arrays representing an image.

    """
    # Unpack target size tuple
    target_width, target_height = target_size

    # Get original dimensions
    original_height, original_width = image.shape[:2]

    # Compute scaling factors
    scale_width = target_width / original_width
    scale_height = target_height / original_height
    scale = min(scale_width, scale_height)

    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a blank canvas for padding
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Compute padding offsets
    y_padding = (target_height - new_height) // 2
    x_padding = (target_width - new_width) // 2

    # Place the resized image in the center of the canvas
    canvas[y_padding:y_padding+new_height, x_padding:x_padding+new_width] = resized_image

    return canvas

def crop_zeros(image, threshold = 20):

    """
        This function removes the 0 values of an np.array representing an image in order
        to crop its size, maintaining only relevant information. It crops out the black
        regions from an image, where the black pixels have a value below the threshold.


        Parameters:
        image: the input image in RGB format (np.array).
        threshold: The value below which a pixel is considered "black". Default is 15.

        Returns:
        cropped_image: the cropped image with black regions removed.

        """

    # Create a mask for pixels close to black (for RGB images)
    black_mask = np.all(image < threshold, axis=-1)

    # Create a mask for the non-black regions (invert the black mask)
    non_black_mask = ~black_mask

    # Find the non-zero pixels in the mask (non-black regions)
    non_zero_indices = np.nonzero(non_black_mask)

    # Find boundaries of non-black image
    y_min, y_max = non_zero_indices[0].min(), non_zero_indices[0].max() + 1
    x_min, x_max = non_zero_indices[1].min(), non_zero_indices[1].max() + 1

    # Crop image using the defined boundaries
    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image


# def preprocess_image(image, target_size, threshold=20):

    """
    Crop non-useful regions and resize the image without distortion.

    Parameters:
    - image (np.array): Input image in RGB format.
    - target_size (tuple): Desired size as (width, height).
    - threshold (int): Pixel intensity threshold for cropping.

    Returns:
    - np.array: Preprocessed image with the target size.
    """

    # Step 1: Crop the image based on the threshold
    black_mask = np.all(image < threshold, axis=-1)
    non_black_mask = ~black_mask
    non_zero_indices = np.nonzero(non_black_mask)

    y_min, y_max = non_zero_indices[0].min(), non_zero_indices[0].max() + 1
    x_min, x_max = non_zero_indices[1].min(), non_zero_indices[1].max() + 1
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Step 2: Resize the cropped image to the target size
    target_width, target_height = target_size
    original_height, original_width = cropped_image.shape[:2]

    # Compute scaling factors
    scale_width = target_width / original_width
    scale_height = target_height / original_height
    scale = min(scale_width, scale_height)

    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize the image
    resized_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a blank canvas for padding
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Compute padding offsets
    y_padding = (target_height - new_height) // 2
    x_padding = (target_width - new_width) // 2

    # Place the resized image in the center of the canvas
    canvas[y_padding:y_padding+new_height, x_padding:x_padding+new_width] = resized_image

    return canvas
