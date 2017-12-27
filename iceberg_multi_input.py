# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 16:03:15 2017

Statoil/C-CORE Iceberg Classifier Challenge (Kaggle)

@author: Gary
"""

import json
import numpy as np
import keras
import csv

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


# Loading dataset

training_data_path = "C:\\Users\\Gary\\iceberg-classifier\\data\\train\\train.json" 

with open(training_data_path, 'r') as f:
    train_data = json.load(f)
    
# Building training data
    
images_train = []
labels_train = []
inc_angle_train = []

angle_data = [image['inc_angle'] for image in train_data]

average_inc_angle_train = np.mean(angle_data[angle_data>0])

for image in train_data:
    current_image_combined = np.reshape((image['band_1'] + image['band_2']), (75, 75, 2)) 
    labels_train.append(image['is_iceberg'])
    images_train.append(current_image_combined)
    
    current_inc_angle = image['inc_angle']
    if type(current_inc_angle) != float: # replaces 'na' with the mean
        inc_angle_train.append(average_inc_angle_train)
    else:
        inc_angle_train.append(current_inc_angle)

images_train = np.array(images_train)
labels_train = np.array(labels_train)
inc_angle_train = np.array(inc_angle_train)

print (images_train[0].shape)

# Building test data


test_data_path = "C:\\Users\\Gary\\iceberg-classifier\\data\\test\\test.json"

with open(test_data_path, 'r') as f:
    test_data = json.load(f)
    
images_test = []
images_id_test = []

for image in test_data:
    current_image_combined = np.reshape((image['band_1'] + image['band_2']), (75, 75, 2))
    images_test.append(current_image_combined)
    images_id_test.append(image['id'])

images_test = np.array(images_test)

print(images_test.shape)

# Define the Model Architecture

main_input = Input(shape=(75,75,2))

conv1 = Conv2D(filters = 16, kernel_size = 2, padding = 'same', activation='relu',
                 input_shape=(75, 75, 2))(main_input)
maxpool1 = MaxPooling2D(pool_size=2)(conv1)
maxpool1 = Dropout(0.2)(maxpool1)

conv2 = Conv2D(filters = 32, kernel_size = 2, padding = 'same', activation = 'relu')(maxpool1)
maxpool2 = MaxPooling2D(pool_size=2)(conv2)
maxpool2 = Dropout(0.2)(maxpool2)

conv3 = Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu')(maxpool2)
maxpool3 = MaxPooling2D(pool_size=2)(conv3)
maxpool3 = Dropout(0.2)(maxpool3)

conv4 = Conv2D(filters = 128, kernel_size = 2, padding = 'same', activation = 'relu')(maxpool3)
maxpool4 = MaxPooling2D(pool_size=2)(conv4)
maxpool4 = Dropout(0.2)(maxpool4)

conv5 = Conv2D(filters = 256, kernel_size = 2, padding = 'same', activation = 'relu')(maxpool4)
maxpool5 = MaxPooling2D(pool_size=2)(conv5)
maxpool5 = Dropout(0.2)(maxpool5)

conv6 = Conv2D(filters = 512, kernel_size = 2, padding = 'same', activation = 'relu')
maxpool6 = MaxPooling2D(pool_size=2)
maxpool6 = Dropout(0.3)(maxpool6)
maxpool6 = Flatten()(maxpool6)

dense1 = Dense(500, activation = 'relu')(maxpool6)
dense1 = Dropout(0.3)(dense1)

auxiliary_input = Input(shape=(1,))

x = keras.layers.concatenate([dense1, auxiliary_input])

x = Dense(64, activation = 'relu')(x)
x = Dense(128, activation = 'relu')(x)

main_output = Dense(1, activation = 'sigmoid')(x)

model = Model(inputs=[main_input, auxiliary_input], output=main_output)

model.compile(optimizer='adam', loss='binary_crossentropy', loss_weights =1.)

model.fit([images_train, inc_angle_train], labels_train, epochs = 100, batch_size = 100)


print (model.summary())

# Compile the Model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the Model

checkpointer = ModelCheckpoint(filepath='model_adam2.weights.best.hdf5', verbose=1, save_best_only=True)

hist = model.fit(images_train, labels_train, batch_size = 100, epochs = 100,
                 validation_split = 0.2,
                 callbacks=[checkpointer], verbose=2, shuffle=True)

# Load model
'''
model.load_weights('model_adam2.weights.best.hdf5')

# Calculate accuracy on test set


predictions = [model.predict(np.expand_dims(image,axis=0))[0][0] for image in images_test] # predictions in "is_iceberg"

print (predictions)

print (len(predictions), len(images_id_test))


with open('submission_adam2.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["id", "is_iceberg"])
    writer.writerows(zip(images_id_test, predictions))
'''

# https://stackoverflow.com/questions/19302612/how-to-write-data-from-two-lists-into-columns-in-a-csv