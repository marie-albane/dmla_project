import numpy as np
from typing import Tuple
from tensorflow import keras
from keras import Model, Sequential, layers, optimizers, metrics
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from dmla.params import TARGETED_IMAGES_X,TARGETED_IMAGES_Y
# from dmla.ml_logic.preprocessor import *


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

    print("✅ Initialisation modèle: done \n")

    return model


def compile_model(model):
    model.compile(loss = 'binary_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy',metrics.Recall(), metrics.Precision()])

    print("✅ Compilation modèle: done \n")
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

def modelisation(X_train_proc,
                 y_train_proc,
                 X_val_proc,
                 y_val_proc,
                 X_test_proc,
                 y_test_proc):

    # Initialiser et compiler le modèle
    model = initialize_model()
    model = compile_model(model)

    # Entraîner le modèle
    model, history = train_model(
        model=model,
        X=X_train_proc,
        y=y_train_proc,
        batch_size=32,
        validation_data=(X_val_proc, y_val_proc)
    )

    print("Entraînement terminé.")

    # Évaluation du modèle
    # resultat = model.evaluate(X_val_proc, y_val_proc, verbose=0)

    # Afficher les métriques
    # print(f"Accuracy sur le jeu de test : {resultat[1]:.2f}")

    # print("Evaluation du modèle:")
    # for i, metric in enumerate(model.metrics_names):
    #     print(f'{metric.capitalize()}: {resultat[i]:.3f}')

    return model, history

if __name__ == '__main__':
    # model = initialize_model()
    # model = compile_model(model)
    pass
