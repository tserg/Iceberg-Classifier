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

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


# Loading dataset

training_data_path = "C:\\Users\\Gary\\iceberg-classifier\\data\\train\\train.json" 

with open(training_data_path, 'r') as f:
    train_data = json.load(f)
    
# Building training data

def build_train_images(data):
    """
    Helper function to build images dataset from json
    
    Parameters:
        data
        
    Returns:
        Array of images
        Array of images labels
    
    """
    
    images = []
    images_labels = []
    
    band1_data = [image['band_1'] for image in train_data]
    band2_data = [image['band_2'] for image in train_data]
    
    # Normalize data : https://www.kaggle.com/vincento/keras-starter-4l-0-1694-lb-icebergchallenge
    
    band1_mean = np.mean(band1_data)
    band1_max = np.max(band1_data)
    band1_min = np.min(band1_data)
    
    band2_mean = np.mean(band2_data)
    band2_max = np.max(band2_data)
    band2_min = np.max(band1_data)
    
    for image in data:
        current_image_band1 = np.reshape((image['band_1']), (75,75))
        current_image_band1 = (current_image_band1-band1_mean) / (band1_max-band1_min)
        
        current_image_band2 = np.reshape((image['band_2']), (75,75))
        current_image_band2 = (current_image_band2 - band2_mean) / (band2_max-band2_min)
        
        current_image_combined = np.reshape(([(band1 + band2)/2 for band1, band2 in zip(current_image_band1, current_image_band2)]), (75,75))
        
        # np.stack to give (75,75,3) shape
        
        current_image = np.stack((current_image_band1,
                                  current_image_band2,
                                  current_image_combined), axis=-1)
    
        images.append(current_image)
        images_labels.append(image['is_iceberg'])
        
    
    return np.array(images), np.array(images_labels)
        
images_train, labels_train = build_train_images(train_data)
print (images_train[0].shape)

# Building test data


test_data_path = "C:\\Users\\Gary\\iceberg-classifier\\data\\test\\test.json"

with open(test_data_path, 'r') as f:
    test_data = json.load(f)
    
def build_test_images(data):
    """
    Helper function to build images dataset from json
    
    Parameters:
        data
        
    Returns:
        Array of images
        Array of images IDs
    
    """
    
    images = []
    images_id = []
    
    band1_data = [image['band_1'] for image in test_data]
    band2_data = [image['band_2'] for image in test_data]
    
    # Normalize data : https://www.kaggle.com/vincento/keras-starter-4l-0-1694-lb-icebergchallenge
    
    band1_mean = np.mean(band1_data)
    band1_max = np.max(band1_data)
    band1_min = np.min(band1_data)
    
    band2_mean = np.mean(band2_data)
    band2_max = np.max(band2_data)
    band2_min = np.max(band1_data)
    
    for image in data:
  
        current_image_band1 = np.reshape((image['band_1']), (75,75))
        current_image_band1 = (current_image_band1-band1_mean) / (band1_max-band1_min)
        
        current_image_band2 = np.reshape((image['band_2']), (75,75))
        current_image_band2 = (current_image_band2 - band2_mean) / (band2_max-band2_min)
        
        current_image_combined = np.reshape(([(band1 + band2)/2 for band1, band2 in zip(current_image_band1, current_image_band2)]), (75,75))
        
        # np.stack to give (75,75,3) shape
        
        current_image = np.stack((current_image_band1,
                                  current_image_band2,
                                  current_image_combined), axis=-1)
        
        images.append(current_image)
        images_id.append(image['id'])
    
    return np.array(images), np.array(images_id)

images_test, images_id_test = build_test_images(test_data)
print(images_test.shape)

# Define the Model Architecture

model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = 2, padding = 'same', activation='relu',
                 input_shape=(75, 75, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 32, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 128, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 256, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 512, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.3))

'''
# Additional dense layers does not improve score

model.add(Dense(500, activation = 'relu')) # additional layer 1
model.add(Dropout(0.3))

model.add(Dense(500, activation = 'relu')) # additional layer 2
model.add(Dropout(0.3))
'''

model.add(Dense(1, activation="sigmoid"))


print (model.summary())

# Compile the Model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the Model

checkpointer = ModelCheckpoint(filepath='model_adam_test.weights.best.hdf5', verbose=1, save_best_only=True)

hist = model.fit(images_train, labels_train, batch_size = 100, epochs = 100,
                 validation_split = 0.2,
                 callbacks=[checkpointer], verbose=2, shuffle=True)

# Load model

model.load_weights('model_adam_test.weights.best.hdf5')

# Calculate accuracy on test set


predictions = [model.predict(np.expand_dims(image,axis=0))[0][0] for image in images_test] # predictions in "is_iceberg"

print (predictions)

print (len(predictions), len(images_id_test))


with open('submission_adam_test.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["id", "is_iceberg"])
    writer.writerows(zip(images_id_test, predictions))


# https://stackoverflow.com/questions/19302612/how-to-write-data-from-two-lists-into-columns-in-a-csv