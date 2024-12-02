#Import à cleaner
from dmla.params import *
from dmla.ml_logic.registry import load_model
from tensorflow import keras
from keras import Model
from dmla.ml_logic.preprocessor import load_and_process_random_image

#faire best_model.predict(image)
def predict() :

    # Charger une image aléatoire preprocessed
    image_test = load_and_process_random_image(wanted_dataset = "testing")

    #Charger le model avec la fonction best_model = load_model() et l image
    best_model = load_model()

    result = best_model.predict(image_test)

    #Message à l'utilisateur. Optionnel : ajouter la probabiblité du model
    if result == 0:
        return "✅ Rassurez-vous, vous n'avez pas la DMLA"
    else :
        return " ⚠️ Vous présentez les symptômes d'une DMLA, nous allons programmer un prochain rendez-vous spécialiste"
