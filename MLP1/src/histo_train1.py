#################################################################

# Pyhton script which plots the hystograms of the 1st training brain

#################################################################

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

data = train1.get_data()

# store all the non zero values in a 1D list
intList=[]
for x in range(X):
    for y in range(Y):
        for z in range(Z):
            if data[x,y,z]!=0:
                intList.append(int(data[x,y,z]))

# plot and save figure
histo = plt.figure(0)
plt.hist(intList, 200)
plt.title("Histogram of train1")
plt.xlabel("Values of the 3D brain")
plt.ylabel("Frequencies")
histo.savefig("../plots/plot_train1_histo.png")
