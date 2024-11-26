import numpy as np
import pandas as pd
import cv2
import os


def resize_without_distortion(image, target_width, target_height):

    """

        Resizes images without distorting them

        Parameters:
        folder_path (str): Path to the folder containing images
        desiderd width (int)
        desired height (int)

        Returns:
        list: list of np.arrays representing an image.

    """

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
