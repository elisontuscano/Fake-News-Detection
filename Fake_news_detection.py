#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 22:52:27 2020

@author: elisontuscano
"""
#importing all libraries
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib

#import Dataset
df=pd.read_csv('news.csv')
df.shape
labels=df.label

#split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

#set up tfidfvectorizor
tfidf_vectorizor=TfidfVectorizer(stop_words='english', max_df=0.7)

#fit and transform train and test set
tfidf_train=tfidf_vectorizor.fit_transform(x_train)
tfidf_test=tfidf_vectorizor.transform(x_test)

#set up PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])

joblib.dump(pac,'model/pac_model.sav')
joblib.dump(tfidf_vectorizor,'model/tfidf_model.sav')
