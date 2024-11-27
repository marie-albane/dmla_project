import os
import numpy as np
from PIL import Image
from pathlib import Path

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
