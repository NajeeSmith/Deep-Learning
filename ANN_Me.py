# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 17:07:28 2018

@author: carto
"""

#Libraries
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Dataset
data = pd.read_csv('Churn_Modelling.csv')
X = data.iloc[:, 3:13].values
y = data.iloc[:, 13].values

#Fixing Categorical Data
L_Encoder_Place = LabelEncoder()
X[:, 1] = L_Encoder_Place.fit_transform(X[:, 1])
L_Encoder_Gender = LabelEncoder()
X[:, 2] = L_Encoder_Gender.fit_transform(X[:, 2])
OHE = OneHotEncoder(categorical_features = [1])
X = OHE.fit_transform(X).toarray()
X = X[:, 1]

#Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

import keras as k
from keras.models import Sequential
from keras.layers import Dense

#Initialization of ANN
classifier = Sequential()

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

prediction = classifier.predict(X_test)
prediction = (prediction > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)