# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 16:03:15 2017

Statoil/C-CORE Iceberg Classifier Challenge (Kaggle)

@author: Gary
"""

import json
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


# Loading dataset

training_data_path = "C:\\Users\\Gary\\iceberg-classifier\\data\\train\\train.json" 

with open(training_data_path, 'r') as f:
    train_data = json.load(f)
    
# Building training data
    
images_train = []
labels_train = []

for image in train_data:
    current_image = np.reshape(image['band_1'], (75, 75, 1))
    labels_train.append(image['is_iceberg'])
    images_train.append(current_image)

num_classes = 2

images_train = np.array(images_train)
labels_train = keras.utils.to_categorical(labels_train, num_classes)

print (images_train[0].shape)

# Building test data

test_data_path = "C:\\Users\\Gary\\iceberg-classifier\\data\\test\\test.json"

with open(test_data_path, 'r') as f:
    test_data = json.load(f)
    
images_test = []
images_id_test = []

for image in test_data:
    current_image = np.reshape(image['band_1'], (75, 75, 1))
    images_test.append(current_image)
    images_id_test.append(image['id'])

images_test = np.array(images_test)

print(images_test.shape)

# Define the Model Architecture

model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = 2, padding = 'same', activation='relu',
                 input_shape=(75, 75, 1)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 32, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.3))

model.add(Dense(2, activation="sigmoid"))


print (model.summary())

# Compile the Model

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train the Model
'''
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)

hist = model.fit(images_train, labels_train, batch_size = 32, epochs = 100,
                 validation_split = 0.5,
                 callbacks=[checkpointer], verbose=2, shuffle=True)
'''

# Load model

model.load_weights('model.weights.best.hdf5')

# Calculate accuracy on test set

predictions = [model.predict(images_test)]
predictions_with_id = list(zip(images_id_test, predictions))

np.savetxt("sample_submission.csv", predictions_with_id, delimiter=",", header = "id,is_iceberg")