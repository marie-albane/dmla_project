import numpy as np
import pandas as pd
import os

from pathlib import Path
from dmla.params import *
from dmla.ml_logic.model import initialize_model, compile_model, train_model, modelisation
from dmla.ml_logic.preprocessor import preprocess_images
from dmla.ml_logic.registry import load_model, save_model, save_results


# préparation des données
X_train_proc, y_train_proc = preprocess_images(path='data/raw_data/training', data_set = "training")
X_val_proc, y_val_proc = preprocess_images(path='data/raw_data/validation',data_set = "validation")
X_test_proc, y_test_proc = preprocess_images(path='data/raw_data/testing',data_set = "testing")



model,history = modelisation(X_train_proc,
                 y_train_proc,
                 X_val_proc,
                 y_val_proc,
                 X_test_proc,
                 y_test_proc):


# Étape 3 : Sauvegarder le modèle et les résultats
#print("Sauvegarde du modèle et des résultats...")
#save_model(model, params)
#save_results(score)

#print("Pipeline terminé avec succès.")


#if __name__ == '__main__':
