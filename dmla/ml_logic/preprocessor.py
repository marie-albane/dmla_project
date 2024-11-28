import numpy as np
import cv2
import os
from PIL import Image
from pathlib import Path


def resize_without_distortion(image, target_size):

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

def crop_zeros(image, threshold = 20):

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


def preprocess_images(image_list, target_size, crop_threshold=20):

    """
    Preprocess a list of np.arrays representing images by cropping and resizing.

    Parameters:
        image_list (list): List of np.arrays representing images.
        target_size (tuple): Target size for resizing (width, height).
        crop_threshold (int): Threshold for cropping black regions.

    Returns:
        list: List of preprocessed np.arrays.
    """

    preprocessed_images = []
    for image in image_list:
        cropped_image = crop_zeros(image, threshold=crop_threshold)
        resized_image = resize_without_distortion(cropped_image, target_size)
        normalized_image = resized_image / 255
        preprocessed_images.append(normalized_image)

    return np.array(preprocessed_images)



def standardisation(relative_path="../data/100_data/training",
                    nb_pixels=255,
                    target_dim=(None, None, 3)):
    """
    Charge les images d'un dossier, redimensionne si nécessaire,
    normalise les pixels et retourne les données.

    Parameters:
        relative_path (str): Chemin relatif vers le dossier contenant les images.
        nb_pixels (int): Valeur pour normaliser les pixels (par défaut 255).
        target_dim (tuple): Dimensions cibles (height, width, channels). None pour ne pas redimensionner.

    Returns:
        np.ndarray: Tableau contenant les valeurs normalisées de toutes les images.
    """
    # Construire le chemin absolu depuis le chemin relatif
    base_dir = Path(__file__).resolve().parent
    # image_dir = os.path.join(base_dir, relative_path)
    image_dir = os.path.join(relative_path)


    # Vérifier que le dossier existe
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Le dossier spécifié n'existe pas : {image_dir}")

    # Liste pour stocker les images prétraitées
    images_data = []

    # Parcourir tous les fichiers du dossier
    for file_name in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file_name)

        # Vérifier si le fichier est une image (par extension)
        if file_name.lower().endswith('.png'):
            # Charger l'image
            with Image.open(file_path) as img:
                # Redimensionner si une taille cible est donnée
                if target_dim[0] is not None and target_dim[1] is not None:
                    img = img.resize((target_dim[1], target_dim[0]))

                # Convertir l'image en tableau numpy
                img_array = np.array(img).astype('float32')

                # Gérer les images en niveaux de gris
                if img_array.ndim == 2:
                    img_array = np.expand_dims(img_array, axis=-1)
                if img_array.shape[-1] != target_dim[-1]:
                    raise ValueError(f"L'image {file_name} ne correspond pas aux dimensions attendues {target_dim}.")

                # Normaliser les pixels
                img_array /= nb_pixels

                # Ajouter au tableau des images
                images_data.append(img_array)

    # Convertir la liste en tableau numpy
    return np.array(images_data)
    # model.py needs to be modified.
