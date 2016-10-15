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

modelLasso = linear_model.Lasso(alpha=1.0)

modelRidge.fit(image,age)
