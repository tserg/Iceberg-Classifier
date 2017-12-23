# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 16:03:15 2017

Statoil/C-CORE Iceberg Classifier Challenge (Kaggle)

@author: Gary
"""

import json

training_data_path = "C:\\Users\\Gary\\iceberg-classifier\\data\\train\\train.json"

with open(training_data_path, 'r') as f:
    train_data = json.load(f)
    
print (train_data[['id'] == "dfd5f913"]['is_iceberg'])
