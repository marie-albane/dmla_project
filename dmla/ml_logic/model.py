import numpy as np
from typing import Tuple
from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from dmla.params import TARGETED_IMAGES_X,TARGETED_IMAGES_Y


def initialize_model():

    input_dim = (TARGETED_IMAGES_X,TARGETED_IMAGES_Y,3)

    model = Sequential()

    model.add(Conv2D(16, (5, 5), activation = 'relu', padding = 'same', input_shape=(TARGETED_IMAGES_X,TARGETED_IMAGES_Y,3)))
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


if __name__=='__main__' :
    model = initialize_model()
