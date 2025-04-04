import numpy as np
import pandas as pd
import random
import cv2
import os
from PIL import Image
from pathlib import Path
from dmla.params import DATA_PATH

def load_X(wanted_dataset="Training", data_path=DATA_PATH):

    """
    Loads all images from a specified dataset folder.

    Parameters:
        wanted_dataset (str): Dataset name, defaults to "Training".
        data_path (str): Base path to the data folder, defaults to environment variable DATA_PATH.

    Returns:
        list: A list of images (np.arrays in RGB format).
    """
    print("Lancement de load_X pour {wanted_dataset} avec le path {data_path}")
    # Normalize dataset name
    wanted_dataset = wanted_dataset.capitalize()

    # Construct path
    images_path = os.path.join(data_path, "raw_data", wanted_dataset.lower())

    # Check if the folder exists
    if not os.path.isdir(images_path):
        raise FileNotFoundError(f"The folder {images_path} does not exist.")

    # Loop through files in the folder

    X = []
    compteur_load = 0


    for file_name in sorted(os.listdir(images_path), key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else float('inf')):
        file_path = os.path.join(images_path, file_name)
        file_name_final = file_name

        # Check if it is a file
        if os.path.isfile(file_path):
            # Read the image
            image = cv2.imread(file_path)
            if image is None:
                print(f"Warning: {file_name} could not be loaded.")
                continue

            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            X.append(image_rgb)
            compteur_load += 1
            print(f"Nombre d'images chargées {compteur_load}")

    return X




def load_y(wanted_dataset="Training", data_path= DATA_PATH):

    """
    This function loads the target (y) as a DataFrame.

    Parameters:
        wanted_dataset (str): Dataset name, defaults to "Training".
        data_path (str): Base path to the data folder, defaults environment variable DATA_PATH

    Returns:
        pd.Series: A Series containing the ARMD target column.
    """
    print("Lancement de load_y sur {wanted_dataset} avec {data_path}")

    # Normalize dataset name
    wanted_dataset = wanted_dataset.capitalize()

    # Construct paths
    images_path = os.path.join(DATA_PATH, "raw_data", wanted_dataset.lower())
    path_csv = os.path.join(DATA_PATH, "raw_data", f"RFMiD_{wanted_dataset}_Labels.csv")

    # Load data
    try:
        data = pd.read_csv(path_csv)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at path: {path_csv}")

    # Set index and extract ARMD column
    if "ID" not in data.columns or "ARMD" not in data.columns:
        raise ValueError("The required columns ('ID', 'ARMD') are missing from the CSV file.")

    data = data.set_index("ID")
    y = data["ARMD"]

    return y





def resize_images(image, target_size):

    """
        This function resizes images without distorting them.

        Parameters:
        image (np.array): the input image in RGB format
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



def crop_images(image, threshold = 20):

    """
        This function removes the 0 values of an np.array representing an image in order
        to crop its size, maintaining only relevant information. It crops out the black
        regions from an image, where the black pixels have a value below the threshold.


        Parameters:
        image: the input image in RGB format (np.array).
        threshold: The value below which a pixel is considered "black". Default is 20.

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



def normalize_images(image):

    """
    Normalizes the pixel values of an image to the range [0, 1].

    Parameters:
    - image (numpy.ndarray): Input image with pixel values in range [0, 255].

    Returns:
    - numpy.ndarray: Normalized image with pixel values in range [0, 1].
    """

    normalized_image = image / 255.0

    return normalized_image



def load_and_process_images(wanted_dataset="Training",
                               data_path=DATA_PATH,
                               target_size=(256, 256),
                               threshold=20):

    """
    This function loads images from a file and processess them as list
    of np.arrays representing images by cropping and resizing.

    Parameters:
        wanted_dataset(str): name of desired dataset to load
        data_path: base path to the data folder, defaults to environment variable DATA_PATH.
        target_size (tuple): Target size for resizing (width, height), it has a default fo 256x256.
        threshold (int): Threshold for cropping black regions, it has a default of 20.

    Returns:
        tuple: two np.arrays, a ndarray representing the images and a ondarray representing the target.
    """
    print("Lancement de load_and_process_images sur {wanted_dataset}")

    X_load = load_X(wanted_dataset)
    print(f"Toutes les images (X) du dossier {wanted_dataset} sont chargées")
    y = load_y(wanted_dataset)
    print(f"Toutes les y du dossier {wanted_dataset} sont chargées")

    X_processed = []
    compteur_preproc = 0

    for image in X_load:
        cropped_images = crop_images(image, threshold)
        resized_images = resize_images(cropped_images, target_size)
        normalized_images = normalize_images(resized_images)
        X_processed.append(normalized_images)
        compteur_preproc += 1
        print("{compteur_preproc} image(s) preproc dans {wanted_dataset}")

    return np.array(X_processed), np.array(y)




######## CHARGEMENT D'UNE IMAGE RANDOM DE RAW DATA POUR LE PREDICT #########

def load_and_process_random_image(wanted_dataset="testing", data_path=DATA_PATH,
                      target_size=(256, 256), threshold=20):
    """
    Loads and processes a random image from a specified dataset folder.

    Parameters:
        wanted_dataset (str): Dataset name, defaults to "testing".
        data_path (str): Base path to the data folder, defaults to environment variable DATA_PATH.
        target_size (tuple): select a desired size for a processed image, default set to 256, 256.
        threshold: select a desired treshold for cropping, default set to 20.

    Returns:
        np.array: The chosen random image in RGB format.
    """
    print(f"Chargement de l'image à analyser dans {wanted_dataset}")

    # Normalize dataset name
    wanted_dataset = wanted_dataset.lower()

    # Construct path
    images_path = os.path.join(data_path, "raw_data", wanted_dataset)

    # Check if the folder exists
    if not os.path.isdir(images_path):
        raise FileNotFoundError(f"The folder {images_path} does not exist.")

    # Get a list of all files in the folder
    image_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]

    if not image_files:
        raise FileNotFoundError(f"No images found in the folder {images_path}.")

    # Choose a random file
    random_file = random.choice(image_files)

    file_path = os.path.join(images_path, random_file)

    # Load the image
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Failed to load the image {random_file}.")

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image
    cropped_image = crop_images(image_rgb, threshold)
    resized_image = resize_images(cropped_image, target_size)
    normalized_image = normalize_images(resized_image)

    print("L'image {random_file} du dossier {wanted_dataset} est preproc")

    return image_rgb, cropped_image, resized_image, normalized_image, random_file


######## CHARGEMENT D'UNE IMAGE DU WEB POUR LE PREDICT #########

def load_and_process_web_image(wanted_dataset="web_images", data_path=DATA_PATH,
                      target_size=(256, 256), threshold=20):
    """
    Loads and processes an image from web browser.

    Parameters:
        wanted_dataset (str): Dataset name, defaults to "web_images".
        data_path (str): Base path to the data folder, defaults to environment variable DATA_PATH.
        target_size (tuple): select a desired size for a processed image, default set to 256, 256.
        threshold: select a desired treshold for cropping, default set to 20.

    Returns:
        np.array: The chosen random image in RGB format.
    """
    print(f"Chargement de l'image à analyser dans {wanted_dataset}")

    # Normalize dataset name
    wanted_dataset = wanted_dataset.lower()

    # Construct path
    images_path = os.path.join(data_path, wanted_dataset)
    print(f"Le dossier de l'image est {images_path}")


    # Check if the folder exists
    if not os.path.isdir(images_path):
        raise FileNotFoundError(f"The folder {images_path} does not exist.")

    # Get a list of all files in the folder - inutile je pense
    valid_extensions = ('.png', '.jpg', '.jpeg')  # Add more extensions if needed
    image_files = [
        os.path.join(images_path, f) for f in os.listdir(images_path)
        if os.path.isfile(os.path.join(images_path, f)) and f.lower().endswith(valid_extensions)
    ]
    print(f"Voici la liste des images {image_files}")

    if not image_files:
        raise FileNotFoundError(f"No images found in the folder {images_path}.")

    # Load the most recent image
    last_file = image_files[0]
    file_name = os.path.basename(last_file)
    file_path = os.path.join(images_path, last_file)


    print(f"Le last_file est {last_file} ---  Le file_name est {file_name} ----- Le file_path sest {last_file}")


    # Load the image
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Failed to load the image in {image_files}.")

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image
    cropped_image = crop_images(image_rgb, threshold)
    resized_image = resize_images(cropped_image, target_size)
    normalized_image = normalize_images(resized_image)

    print("L'image {random_file} du dossier {wanted_dataset} est preproc")

    return image_rgb, cropped_image, resized_image, normalized_image, file_name




##### POUR ENTRAINEMENT ########

def load_100_X(wanted_dataset="Training", data_path=DATA_PATH):

    """
    Loads all images from a specified dataset folder.

    Parameters:
        wanted_dataset (str): Dataset name, defaults to "Training".
        data_path (str): Base path to the data folder, defaults to environment variable DATA_PATH.

    Returns:
        list: A list of images (np.arrays in RGB format).
    """

    # Normalize dataset name
    wanted_dataset = wanted_dataset.capitalize()

    # Construct path
    images_path = os.path.join(data_path, "100_data", wanted_dataset.lower())

    # Check if the folder exists
    if not os.path.isdir(images_path):
        raise FileNotFoundError(f"The folder {images_path} does not exist.")

    # Loop through files in the folder
    X_100 = []

    for file_name in sorted(os.listdir(images_path), key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else float('inf')):
        file_path = os.path.join(images_path, file_name)

        # Check if it is a file
        if os.path.isfile(file_path):
            # Read the image
            image = cv2.imread(file_path)
            if image is None:
                print(f"Warning: {file_name} could not be loaded.")
                continue

            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            X_100.append(image_rgb)

    return X_100


def load_100_y(wanted_dataset="Training", data_path= DATA_PATH):

    # Normalize dataset name
    wanted_dataset = wanted_dataset.capitalize()

    # Construct paths
    images_path_100 = os.path.join(data_path, "100_data", wanted_dataset.lower())
    path_csv= os.path.join(DATA_PATH, "raw_data", f"RFMiD_{wanted_dataset}_Labels.csv")

    # Load data
    try:
        data = pd.read_csv(path_csv)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at path: {path_csv}")

    # Set index and extract ARMD column
    if "ID" not in data.columns or "ARMD" not in data.columns:
        raise ValueError("The required columns ('ID', 'ARMD') are missing from the CSV file.")

    nb_sample = int(len(os.listdir(images_path_100))/2)
    data1 = data[data["ARMD"]==1].head(nb_sample)["ARMD"]
    data2 = data[data["ARMD"]==0].head(nb_sample)["ARMD"]
    y_100 = pd.concat([data1, data2], ignore_index=True)

    return y_100




def load_and_process_100images(wanted_dataset="Training",
                               data_path=DATA_PATH,
                               target_size=(256, 256),
                               threshold=20):

    """
    This function loads sample images from a file and processess them as list
    of np.arrays representing images by cropping and resizing.

    Parameters:
        wanted_dataset(str): name of desired dataset to load
        data_path: base path to the data folder, defaults to environment variable DATA_PATH.
        target_size (tuple): Target size for resizing (width, height), it has a default fo 256x256.
        threshold (int): Threshold for cropping black regions, it has a default of 20.

    Returns:
        tuple: two np.arrays, a ndarray representing the images and a ondarray representing the target.
    """

    X_load = load_100_X(wanted_dataset)
    y_100 = load_100_y(wanted_dataset)

    X_100_processed = []

    for image in X_load:
        cropped_images = crop_images(image, threshold)
        resized_images = resize_images(cropped_images, target_size)
        normalized_images = normalize_images(resized_images)
        X_100_processed.append(normalized_images)

    return np.array(X_100_processed), np.array(y_100)
