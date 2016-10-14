########################################################################

# Python script to output result from ridge regression

########################################################################

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import linear_model

#import sklearn as sk
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


reg = linear_model.Ridge (alpha = [0.5,0.5,0.5])
linear_model.RidgeCV(alphas=[0.1])
clf.fit(X,labels) # labels are the age
score = sklearn.cross_validation.cross_val_score()
output_data= clf.predict(X_t).astype(int)
