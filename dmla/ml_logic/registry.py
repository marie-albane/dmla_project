import glob
import os
import time
import pickle

from colorama import Fore, Style
from tensorflow import keras
from dmla.params import DATA_PATH



def save_model(model: keras.Model = None) -> None:
    """
    Pour l'instant : uniquement stocker 1 seul modèle en local dans data

    Optimisation à faire:
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only

    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(DATA_PATH, "models",f"{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    return None


def load_model() -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    # Get the latest model version name by the timestamp on disk
    local_model_directory = os.path.join(DATA_PATH, "models")
    local_model_paths = glob.glob(f"{local_model_directory}/*")

    if not local_model_paths:
        return None

    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    latest_model = keras.models.load_model(most_recent_model_path_on_disk)

    model_number = most_recent_model_path_on_disk.split('/')[-1].split('.')[0]

    print(f"✅ Vous venez de charger le dernier modèle, le n° {model_number}")

    return latest_model, model_number
