########################################################################

# Python script to output result from ridge regression

########################################################################

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn as sk
#from sklearn.linear_model import Lasso

import os, sys

#'''
X = 175
Y = 207
Z = 175

''' #for testing
X = 10
Y = 10
Z = 10
'''

#relative path
#train1 = nib.load("../data/set_train/train_1.nii")

#path from usb key
train1 = nib.load("/run/media/lionelt/04F6-B693/ML/data/set_train/train_1.nii")
data = train1.get_data()

# RIDGE REGRESSION

import sklearn

# More info at :
# http://scikit-learn.org/stable/tutorial/basic/tutorial.html
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score

# We choose several value of alpha
modelRidge = linear_model.RidgeCV(alphas=[0.1, 0.2, 0.3])

# We compute the model
modelRidge.fit(X,Y)

# Cross validation
# we compute cv times the score with different splits each time (partioning)
score = sk.cross_validation.cross_val_score(modelRidge, datatofit, targetvariable, cv=5)

# Prediction
# toPredict = images to predict
predicteddata= modelRidge.predict(toPredict).astype(int)