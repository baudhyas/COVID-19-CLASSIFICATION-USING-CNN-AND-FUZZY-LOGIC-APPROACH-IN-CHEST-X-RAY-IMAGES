#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image

def main():
    TRAIN_PATH = Path('CovidDataset/Train').absolute()
    VAL_PATH = Path('CovidDataset/Test').absolute()

    # CNN Based Model in Keras

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    model.summary()


    train_datagen = image.ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
    )

    test_dataset = image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'CovidDataset/Train/',
        target_size=(224, 224),
        batch_size = 32,
        class_mode = 'binary'
    )

    validation_generator = test_dataset.flow_from_directory(
        'CovidDataset/Val',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    hist = model.fit_generator(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=2
    )

    model.save('model.keras')




