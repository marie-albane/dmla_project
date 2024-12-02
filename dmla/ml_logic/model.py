import numpy as np
from typing import Tuple
from tensorflow import keras
from keras import Model, Sequential, layers, optimizers, metrics
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from dmla.params import TARGETED_IMAGES_X,TARGETED_IMAGES_Y, DATA_PATH
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
                 y_val_proc):

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

    print("✅ Entrainement modèle: done \n")
    print(model.summary())

    train_size = len(X_train_proc)
    validation_size = len(X_val_proc)

    params = dict(
        context="train",
        data_path = DATA_PATH,
        targeted_image_x=TARGETED_IMAGES_X,
        targeted_image_y=TARGETED_IMAGES_Y,
        train_size=train_size,
        validation_size = validation_size)

    print("======= PARAMETRES =========")
    print("context: ","train")
    print("data path: ",DATA_PATH)
    print(f"taille de l'image: {TARGETED_IMAGES_X}x{TARGETED_IMAGES_X}")
    print("Nombre d images de training: ",train_size)
    print("Nombre d images de validation: ",validation_size)
    print("")

    loss = np.min(history.history['loss'])
    accuracy = np.max(history.history['accuracy'])
    recall = np.max(history.history['recall'])
    precision = np.max(history.history['precision'])
    val_loss = np.min(history.history['val_loss'])
    val_accuracy = np.max(history.history['val_accuracy'])
    val_recall = np.max(history.history['val_recall'])
    val_precision = np.max(history.history['val_precision'])

    print("======= METRIQUES (training / validation) =========")
    print(f"loss: {round(loss,4)} / {round(val_loss,4)}")
    print(f"accuracy: {round(accuracy,4)} / {round(val_accuracy,4)}")
    print(f"recall: {round(recall,4)} / {round(val_recall,4)}")
    print(f"precision: {round(precision,4)} / {round(val_precision,4)}")
    print("")

    metrics_dic = dict(loss=loss,
                    accuracy = accuracy,
                    recall = recall,
                    precision = precision,
                    val_loss = val_loss,
                    val_accuracy = val_accuracy,
                    val_recall = val_recall,
                    val_precision = val_precision)

    # Retourner le modèle et l'historique
    return model, history, params, metrics_dic


if __name__ == '__main__':
    # model = initialize_model()
    # model = compile_model(model)
    pass
