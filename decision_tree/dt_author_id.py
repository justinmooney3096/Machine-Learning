#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
import sys
import os
emailpath = os.path.join(os.path.dirname(os.path.dirname( __file__ )), 'tools')
classvis = os.path.join(os.path.dirname(os.path.dirname( __file__ )), 'choose_your_own')
sys.path.append(emailpath)
sys.path.append(classvis)

from time import time
from class_vis import prettyPicture, output_image
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###


#########################################################


