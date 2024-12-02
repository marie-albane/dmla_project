#Import à cleaner
from dmla.params import *
from dmla.ml_logic.registry import load_model
from tensorflow import keras
from keras import Model
from dmla.ml_logic.preprocessor import load_and_process_random_image
import matplotlib.pyplot as plt

#faire best_model.predict(image)
def predict() :

    # Charger une image aléatoire preprocessed
    image_rgb, normalized_image, image_name = load_and_process_random_image(wanted_dataset = "testing")
    image_with_batch = np.expand_dims(normalized_image, axis=0)
    print("✅ => image chargée")


    #Charger le model avec la fonction best_model = load_model() et l image
    best_model = load_model()
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
    plt.imshow(image_rgb)
    plt.imshow(normalized_image)
    plt.show()

if __name__ == '__main__':
    predict()
