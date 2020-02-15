#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 23:59:41 2020

@author: elisontuscano
"""
#importing libraries
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib


#load model
pac = joblib.load('model/pac_model.sav')
tfidf = joblib.load('model/tfidf_model.sav')

x=input('Enter News : ')

#set up tfidfvectorizor
tfidf_vectorizor=TfidfVectorizer(stop_words='english', max_df=0.7)

#fit and transform train and test set
tfidf_test=tfidf.transform([x,])

y_pred=pac.predict(tfidf_test)


print("The news is %s"%(str(y_pred).strip("['']")))