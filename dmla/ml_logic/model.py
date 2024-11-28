import numpy as np
from typing import Tuple
from tensorflow import keras
from keras import Model, Sequential, layers, optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from dmla.params import TARGETED_IMAGES_X,TARGETED_IMAGES_Y
from dmla.ml_logic.standardisation import standardisation




def initialize_model():

    input_dim = (TARGETED_IMAGES_X,TARGETED_IMAGES_Y,3)

    model = Sequential()

    model.add(Conv2D(16, (5, 5), activation = 'relu', padding = 'same', input_shape=input_dim))
    model.add(MaxPooling2D((2, 2)))


    model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D((2, 2)))


    model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(100, activation = 'relu'))

    model.add(Dense(1, activation = 'sigmoid'))

    return model




def compile_model(model):
    model.compile(loss = 'binary_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    return model



def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=256,
        patience=2,
        validation_data=None, # overrides validation_split
        validation_split=0.3
        ):

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    return model, history


if __name__ == '__main__':

    # Charger les données
    data = standardisation(
        relative_path="data/100_data/training",
        nb_pixels=255,
        target_dim=(TARGETED_IMAGES_X, TARGETED_IMAGES_Y, 3)
    )

    # Simuler des étiquettes pour la classification binaire (par exemple)
    # Vous devrez remplacer ceci par vos vraies étiquettes
    y = np.random.randint(0, 2, size=(data.shape[0],))

    # Diviser en jeu d'entraînement et jeu de test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

    # Initialiser et compiler le modèle
    model = initialize_model()
    model = compile_model(model)

    # Entraîner le modèle
    model, history = train_model(
        model=model,
        X=X_train,
        y=y_train,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    print("Entraînement terminé.")
    print(f"Accuracy sur le jeu de test : {model.evaluate(X_test, y_test)[1]:.2f}")



    # Évaluation du modèle
    res = model.evaluate(X_test, y_test, verbose=0)

    # Nombre de classes pour le calcul du niveau de chance
    num_classes = len(set(y_test))  # ou utilisez len(labels) si labels est une liste des classes
    chance_level = 1. / num_classes * 100

    # Afficher les métriques
    print("Model Evaluation:")
    for i, metric in enumerate(model.metrics_names):
        print(f'{metric.capitalize()}: {res[i]:.3f}')

    # Comparer l'exactitude au niveau de chance
    print(f'The model accuracy is {res[1]*100:.3f}% '
        f'compared to a chance level of {chance_level:.3f}%.')
