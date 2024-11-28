import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil
from dmla.params import *

def plot_images(original_image, preproc_image):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_image)
    axes[0].axis("off")  # Turn off the axis
    axes[0].set_title("Original image")  # Set a title

    # Show the second image on the second subplot
    axes[1].imshow(preproc_image)
    axes[1].axis("off")  # Turn off the axis
    axes[1].set_title("Preprocessed image")  # Set a title

    # Display the images
    plt.tight_layout()  # Adjust spacing between subplots

    return plt.show()




def copier_coller(nb_images_par_classe=100,set_data="training",): #Fait le 26.11.2024 - Michel


    """Fonction pour copier/coller des images depuis le training set:
    - Préparation du nombre d'images à séléctionner, des chemins source et cible
    - Vérification de la création des chemins
    - Copie/colle"""



    #Choix du set de données
    path = os.path.join("data","raw_data","RFMiD_"+set_data.title()+"_Labels.csv")
    #path_training_set = (os.getcwd(),"data\raw_data\RFMiD_Training_Labels.csv") #regarder pour les slachs
    data = pd.read_csv(path)
    data = data.set_index("ID")

    #Dossier source training
    dossier_source = os.path.join(os.getcwd(), "data","raw_data",set_data.lower())

    #Dossier cible training
    dossier_cible = os.path.join(os.getcwd(), "data","100_data",set_data.lower())

    #Lister noms des images avec DLMLA
    positive_set = data[data["ARMD"]==1].head(nb_images_par_classe)
    positive_list = list(positive_set.index)

    #Lister noms des images avec DLMLA
    negative_set = data[data["ARMD"]==0].head(nb_images_par_classe)
    negative_list = list(negative_set.index)

    #Réunir les 2 listes
    images_list = positive_list + negative_list


    # Compteur du nb images
    compteur = 0
    introuvable = ""

    # Vérifier si le dossier cible existe, sinon le créer
    if not os.path.exists(dossier_cible):
        os.makedirs(dossier_cible)

    # Parcourir la liste des noms de fichiers
    for nom_fichier in images_list:
        # Construire le chemin complet des fichiers source et cible
        chemin_source = os.path.join(dossier_source, f"{nom_fichier}.png")
        chemin_cible = os.path.join(dossier_cible, f"{nom_fichier}.png")

        # Vérifier si le fichier existe dans le dossier source
        if os.path.exists(chemin_source):
            # Copier le fichier dans le dossier cible
            shutil.copy(chemin_source, chemin_cible)
            compteur += 1
            #print(f"Fichier copié : {chemin_source} -> {chemin_cible}")
        else:
            introuvable += {chemin_source}

    print(f"✅Total d'images copiées : {compteur} / {len(images_list)} pour {set_data}")
    if len(introuvable)!=0:
        print(f"Attention voici la liste des images non-copiées {introuvable}")

if __name__ == '__main__':
    copier_coller(50,"validation")
