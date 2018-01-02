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
import cv2

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint


# Loading dataset

training_data_path = "C:\\Users\\Gary\\iceberg-classifier\\data\\train\\train.json" 
test_data_path = "C:\\Users\\Gary\\iceberg-classifier\\data\\test\\test.json"

with open(training_data_path, 'r') as f:
    train_data = json.load(f)
    
with open(test_data_path, 'r') as g:
    test_data = json.load(g)
    
# Normalising values

band1_train_data = [image['band_1'] for image in train_data]
band2_train_data = [image['band_2'] for image in train_data]

band1_test_data = [image['band_1'] for image in test_data]
band2_test_data = [image['band_2'] for image in test_data]
    
    # Normalize data : https://www.kaggle.com/vincento/keras-starter-4l-0-1694-lb-icebergchallenge
    
band1_mean = np.mean(band1_train_data)
band1_max = np.max(band1_train_data)
band1_min = np.min(band1_train_data)

band2_mean = np.mean(band2_train_data)
band2_max = np.max(band2_train_data)
band2_min = np.min(band2_train_data)
    
# Building training data

def rotate_image(image, angle):
    """
    Helper function to rotate images
    
    Parameters:
        image: np.array of image
        angle: angle of rotation
            
    Returns:
        dst: rotated np.array of image
    
    """
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    dst = cv2.warpAffine(image, M, (cols, rows))
    
    return dst
    
def build_train_images(data, band_1_mean, band_1_max, band_1_min, band_2_mean, band_2_max, band_2_min):
    """
    Helper function to build images dataset from json
    
    Parameters:
        data: json set of images
        
    Returns:
        Array of images
        Array of images labels
    
    """
        
    images = []
    images_labels = []
    
    # Normalize data : https://www.kaggle.com/vincento/keras-starter-4l-0-1694-lb-icebergchallenge
    
    for image in data:
        current_image_band1 = np.reshape((image['band_1']), (75,75))
        current_image_band1 = (current_image_band1-band_1_mean) / (band_1_max-band_1_min)
        
        current_image_band2 = np.reshape((image['band_2']), (75,75))
        current_image_band2 = (current_image_band2 - band_2_mean) / (band_2_max-band_2_min)
        
        current_image_combined = np.reshape(([(band1 + band2)/2 for band1, band2 in zip(current_image_band1, current_image_band2)]), (75,75))
        
        # flip images horizontally for additional data
        
        flipped_image_horizontal_band1 = cv2.flip(current_image_band1, 0)
        flipped_image_horizontal_band2 = cv2.flip(current_image_band2, 0)
        flipped_image_horizontal_combined = np.reshape(([(band1 + band2)/2 for band1, band2 in zip(flipped_image_horizontal_band1, flipped_image_horizontal_band2)]), (75,75))
        
        flipped_image_vertical_band1 = cv2.flip(current_image_band1, 1)
        flipped_image_vertical_band2 = cv2.flip(current_image_band2, 1)
        flipped_image_vertical_combined = np.reshape(([(band1 + band2)/2 for band1, band2 in zip(flipped_image_vertical_band1, flipped_image_vertical_band2)]), (75,75))
        
        flipped_image_band1 = cv2.flip(current_image_band1, -1)
        flipped_image_band2 = cv2.flip(current_image_band2, -1)
        flipped_image_combined = np.reshape(([(band1 + band2)/2 for band1, band2 in zip(flipped_image_band1, flipped_image_band2)]), (75,75))
        
        # rotate images for additional data
        
        rotated_image_90_band1 = rotate_image(current_image_band1, 90)
        rotated_image_90_band2 = rotate_image(current_image_band2, 90)
        rotated_image_90_combined = np.reshape(([(band1 + band2)/2 for band1, band2 in zip(rotated_image_90_band1, rotated_image_90_band2)]), (75,75))
        
        rotated_image_270_band1 = rotate_image(current_image_band1, 270)
        rotated_image_270_band2 = rotate_image(current_image_band2, 270)
        rotated_image_270_combined = np.reshape(([(band1 + band2)/2 for band1, band2 in zip(rotated_image_270_band1, rotated_image_270_band2)]), (75,75))
        
        # np.stack to give (75,75,3) shape
        
        current_image = np.stack((current_image_band1,
                                  current_image_band2,
                                  current_image_combined), axis=-1)
        
        flipped_image_horizontal = np.stack((flipped_image_horizontal_band1,
                                             flipped_image_horizontal_band2,
                                             flipped_image_horizontal_combined), axis=-1)
        
        flipped_image_vertical = np.stack((flipped_image_vertical_band1,
                                           flipped_image_vertical_band2,
                                           flipped_image_vertical_combined), axis=-1)

        flipped_image = np.stack((flipped_image_band1,
                                  flipped_image_band2,
                                  flipped_image_combined), axis=-1)
        
        rotated_image_90 = np.stack((rotated_image_90_band1,
                                     rotated_image_90_band2,
                                     rotated_image_90_combined), axis = -1)
        
        rotated_image_270 = np.stack((rotated_image_270_band1,
                                      rotated_image_270_band2,
                                      rotated_image_270_combined), axis=-1)
        
        images.append(current_image)
        images.append(flipped_image)
        images.append(flipped_image_horizontal)
        images.append(flipped_image_vertical)
        images.append(rotated_image_90)
        images.append(rotated_image_270)
        
        images_labels.append(image['is_iceberg'])
        images_labels.append(image['is_iceberg'])
        images_labels.append(image['is_iceberg'])
        images_labels.append(image['is_iceberg'])
        images_labels.append(image['is_iceberg'])
        images_labels.append(image['is_iceberg'])
        
    # Shuffle the images because now every 4 images are the same data: significant improvement on val_loss
    
    images_with_labels = list(zip(images, images_labels))
    np.random.shuffle(images_with_labels)
    images, images_labels = zip(*images_with_labels)
        
    return np.array(images), np.array(images_labels)
        
images_train, labels_train = build_train_images(train_data, band1_mean, band1_max, band1_min, band2_mean, band2_max, band2_min)
print (images_train.shape)

# Building test data
    
def build_test_images(data, band_1_mean, band_1_max, band_1_min, band_2_mean, band_2_max, band_2_min):
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
    
    for image in data:
  
        current_image_band1 = np.reshape((image['band_1']), (75,75))
        current_image_band1 = (current_image_band1-band_1_mean) / (band_1_max-band_1_min)
        
        current_image_band2 = np.reshape((image['band_2']), (75,75))
        current_image_band2 = (current_image_band2 - band_2_mean) / (band_2_max-band_2_min)
        
        current_image_combined = np.reshape(([(band1 + band2)/2 for band1, band2 in zip(current_image_band1, current_image_band2)]), (75,75))
        
        # np.stack to give (75,75,3) shape
        
        current_image = np.stack((current_image_band1,
                                  current_image_band2,
                                  current_image_combined), axis=-1)
        
        images.append(current_image)
        images_id.append(image['id'])
    
    return np.array(images), np.array(images_id)

images_test, images_id_test = build_test_images(test_data, band1_mean, band1_max, band1_min, band2_mean, band2_max, band2_min)



print(images_test.shape)

# Define the Model Architecture

model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = 2, padding = 'same', activation='relu',
                 input_shape=(75, 75, 3)))
model.add(Conv2D(filters = 16, kernel_size = 2, padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 32, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 32, kernel_size = 2, padding = 'same', activation = 'tanh'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 128, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 128, kernel_size = 2, padding = 'same', activation = 'tanh'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 256, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 256, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 512, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 512, kernel_size = 2, padding = 'same', activation = 'tanh'))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.3))


# Additional dense layers does not improve score

model.add(Dense(500, activation = 'relu')) # additional layer 1
model.add(Dropout(0.3))

model.add(Dense(500, activation = 'relu')) # additional layer 2
model.add(Dropout(0.3))


model.add(Dense(1, activation="sigmoid"))


print (model.summary())

# Compile the Model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the Model

checkpointer = ModelCheckpoint(filepath='model_01012018_1.weights.best.hdf5', verbose=1, save_best_only=True)

hist = model.fit(images_train, labels_train, batch_size = 128, epochs = 100,
                 validation_split = 0.3,
                 callbacks=[checkpointer], verbose=2, shuffle=True)

# Load model

model.load_weights('model_01012018_1.weights.best.hdf5')

# Calculate accuracy on test set


predictions = [model.predict(np.expand_dims(image,axis=0))[0][0] for image in images_test] # predictions in "is_iceberg"

print (predictions)

print (len(predictions), len(images_id_test))


with open('submission_01012018_1.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["id", "is_iceberg"])
    writer.writerows(zip(images_id_test, predictions))


# https://stackoverflow.com/questions/19302612/how-to-write-data-from-two-lists-into-columns-in-a-csv