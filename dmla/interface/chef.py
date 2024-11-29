import numpy as np
import pandas as pd
import os

from pathlib import Path
from dmla.params import *
from dmla.ml_logic.model import initialize_model, compile_model, train_model, modelisation
from dmla.ml_logic.preprocessor import load_and_process_100images
from dmla.ml_logic.registry import load_model, save_model, save_results


# préparation des données
X_train_proc, y_train_proc = load_and_process_100images(wanted_dataset = "training")
X_val_proc, y_val_proc = load_and_process_100images(wanted_dataset = "validation")
X_test_proc, y_test_proc = load_and_process_100images(wanted_dataset =  "testing")


model, history, params, metrics_dic = modelisation(X_train_proc,
                                                y_train_proc,
                                                X_val_proc,
                                                y_val_proc,
                                                X_test_proc,
                                                y_test_proc)


# Étape 3 : Sauvegarder le modèle et les résultats
#print("Sauvegarde du modèle et des résultats...")
save_model(model)
# save_results(score)

#print("Pipeline terminé avec succès.")


#if __name__ == '__main__':
