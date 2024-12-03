#Import à cleaner
from dmla.params import *
from dmla.ml_logic.registry import load_model
from tensorflow import keras
from keras import Model
from dmla.ml_logic.preprocessor import load_and_process_random_image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path

#faire best_model.predict(image)
def predict() :

    # Charger une image aléatoire preprocessed
    image_rgb, cropped_image, resized_image, normalized_image, image_name = load_and_process_random_image(wanted_dataset = "testing")
    image_with_batch = np.expand_dims(normalized_image, axis=0)
    print("✅ => image chargée")

    #Récupérer le y_true
    images_path = os.path.join(DATA_PATH, "raw_data", "testing")
    path_csv = os.path.join(DATA_PATH, "raw_data", f"RFMiD_Testing_Labels.csv")

    # Load data
    try:
        data = pd.read_csv(path_csv)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at path: {path_csv}")

    # Set index and extract ARMD column
    if "ID" not in data.columns or "ARMD" not in data.columns:
        raise ValueError("The required columns ('ID', 'ARMD') are missing from the CSV file.")

    print(f"image name est {image_name}")
    # Retirer l'extension .png ou autre
    image_base_name = os.path.splitext(image_name)[0]
    data = data.set_index("ID")
    image_index = int(image_base_name) + 1
    y_true = data["ARMD"][image_index]


    #Charger le model avec la fonction best_model = load_model() et l image
    best_model, model_number = load_model()
    print("✅ => modèle chargé")

    result = best_model.predict(image_with_batch)


    #Message à l'utilisateur. Optionnel : ajouter la probabiblité du model
    print("\n==========================")
    print("=        RESULTAT        =")
    print("==========================\n")
    print(f"prédiction de DMLA: {result[0][0]*100:.4} %\n")

    if result < 0.5:
        print(f"✅ Rassurez-vous, vous avez peu de chance d'avoir la DMLA")
    else :
        print(f"⚠️ Vous présentez les symptômes d'une DMLA\n nous allons programmer un prochain rendez-vous spécialiste")

    print("\nPour l'image:",image_name)
    print("basé sur le model: " + model_number)

    if y_true == 1 :
        print(f"Confirmation avec le fichier csv pour l'image {image_base_name} : Présence de  DMLA ⚠️")
    else :
        print(f"Confirmation avec le fichier csv pour l'image {image_base_name}: Absence DMLA✅")

    plt.figure(figsize=(13,5))
    # First subplot
    plt.subplot(1,4,1)
    plt.imshow(image_rgb)
    plt.title('image originale')
    # Second subplot
    plt.subplot(1,4,2)
    plt.imshow(cropped_image)
    plt.title('image croppée (réduction du noir sur les cotés)')
    # Global figure methods
    plt.subplot(1,4,3)
    plt.imshow(resized_image)
    plt.title('image resizée à '+str(TARGETED_IMAGES_X)+'x'+str(TARGETED_IMAGES_Y))

    plt.subplot(1,4,4)
    plt.imshow(normalized_image)
    plt.title('image finale (div des couleurs par 255)')
    plt.show()

    plt.show()

if __name__ == '__main__':
    predict()
