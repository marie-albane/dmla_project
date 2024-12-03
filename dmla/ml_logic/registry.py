import glob
import os
import time

from colorama import Fore, Style
from tensorflow import keras
from dmla.params import DATA_PATH, BUCKET_NAME
from google.cloud import storage


def save_model(model: keras.Model = None) -> None:
    """
    Pour l'instant : uniquement stocker 1 seul modèle en local dans data

    Optimisation à faire:
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only

    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(DATA_PATH, "models",f"{timestamp}.h5")
    model.save(model_path)
    print("✅ Model saved locally")

    # Save model TO GCS

    model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{model_filename}") #Création d'un blop = fichier pour GCS
    blob.upload_from_filename(model_path)
    print("✅ Model saved to GCS")

    return None



def load_model() -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - from GCS (most recent one) --> for unit 02 only (?)

    Return None (but do not Raise) if no model is found
    """
    if MODEL_TARGET == "local":

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(DATA_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        model_number = most_recent_model_path_on_disk.split('/')[-1].split('.')[0]

        print(f"✅ Chargement en local du dernier modèle, le n° {model_number}")

        return latest_model, model_number

    elif MODEL_TARGET == "gcs":

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(DATA_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = keras.models.load_model(latest_model_path_to_save)

            model_number = latest_blob.name.split('/')[-1].split('.')[0]

            print("✅ Latest model downloaded from cloud storage")

            return latest_model, model_number
        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")
            return None, None

    else:
        return None, None
