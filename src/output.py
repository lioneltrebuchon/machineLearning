########################################################################

# Python script which plots all the brains projected onto the 3 plans

########################################################################

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt

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
train1 = nib.load("../data/set_train/train_1.nii")

#path from usb key
#train1 = nib.load("/run/media/lionelt/04F6-B693/ML/data/set_train/train_1.nii")
data = train1.get_data()

ages = np.genfromtxt('../data/targets.csv', delimiter="\n")

sys.stdout.write()