#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import os
import sys
import os
emailpath = os.path.join(os.path.dirname(os.path.dirname( __file__ )), 'tools')
classvis = os.path.join(os.path.dirname(os.path.dirname( __file__ )), 'choose_your_own')
sys.path.append(emailpath)
sys.path.append(classvis)

import pandas as pd
import numpy as np
from time import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from email_preprocess import preprocess



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

## Lowers accuracy having smaller training sets
features_train = features_train[0:100]
labels_train = labels_train[0:100]

#########################################################
### your code goes here ###

clf = SVC(kernel='rbf',C=10000,gamma='auto')#,gamma=1,C=10000) #gamma = 1000, C=1000, kernel=linear gives most accurate results
t0 = time()
clf.fit(features_train,labels_train)
print("Training time:",round(time()-t0,3),'s')

t0 = time()
predict = pd.Series(clf.predict(features_test))
chris_predict = predict.loc[predict == 1]
print('Number predicted emails from Chris:',len(chris_predict))
print("Prediction time:",round(time()-t0,3),'s')

accuracy = accuracy_score(labels_test,predict)
print('Accuracy:',accuracy)


#########################################################


