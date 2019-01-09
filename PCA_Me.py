# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 13:55:30 2018

@author: carto
"""
#Importing Data and Creating Variables
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as SS
data = pd.read_csv('Wine.csv')
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values

#Splitting Training and Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)

#Scaling
#X_train = SS.fit_transform(X_train)
#X_test = SS.fit_transform(X_test)

#Discovering the most important variables using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)

explained_variance = pca.explained_variance_ratio_

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

