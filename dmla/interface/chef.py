import numpy as np
import pandas as pd
import os

from pathlib import Path
from dmla.params import *
from dmla.ml_logic.model import initialize_model, compile_model, train_model, modelisation
from dmla.ml_logic.preprocessor import load_and_process_100images, load_and_process_images
from dmla.ml_logic.registry import load_model, save_model


# préparation des données

X_train_proc, y_train_proc = load_and_process_images(wanted_dataset = "training")
print("Chargement et preprocess terminés pour les images du dossier TRAINING")
X_train_pos, y_train_pos = load_and_process_images(wanted_dataset = "positive")
print("Chargement et preprocess terminés pour les images du dossier POSITIVE ")
X_val_proc, y_val_proc = load_and_process_images(wanted_dataset = "validation")
print("Chargement et preprocess terminés pour les images du dossier VALIDATION ")
#X_test_proc, y_test_proc = load_and_process_images(wanted_dataset =  "testing")

# concaténation du train set

y_train = np.concatenate((y_train_proc, y_train_pos), axis = 0)
X_train = np.concatenate((X_train_proc, X_train_pos), axis = 0)
print("Concaténation des X et y  des dossiers TRAINING et POSITIVE ")

model, history, params, metrics_dic = modelisation(X_train,
                                                y_train,
                                                X_val_proc,
                                                y_val_proc)
print("Fin de la modélisation")


# Étape 3 : Sauvegarder le modèle et les résultats
save_model(model)
print("Sauvegarde de modèle depuis chef ")

# Étape 4 : Charger le modèle (en local uniquement pour l'instant)
best_model = load_model()
print("Chargement du modèle depuis chef ")

print("Pipeline terminé avec succès !")

#if __name__ == '__main__':
