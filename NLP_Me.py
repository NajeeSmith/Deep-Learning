# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 00:31:53 2018

@author: carto
"""
#Import libraries
import pandas as pd
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
data = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#Creating a loop that cleans each text file then collects the cleaned data
corpus = [] #Collection of the individual reviews
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])
    review = review.lower()
    review = review.split()
    stem = PorterStemmer()
    review = [stem.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review =' '. join(review)
    corpus.append(review)
    
#Bag of Words Model via tokenization
from sklearn.feature_extraction.text import CountVectorizer

vector = CountVectorizer(max_features = 1500)
X = vector.fit_transform(corpus).toarray()
y = data.iloc[:, 1].values

#Splitting data into training and testing
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


#Creating NB model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Prediction time!
guess = classifier.predict(X_test)

#Confusion Matrix to determine how many are correct
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, guess)