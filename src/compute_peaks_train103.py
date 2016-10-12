#################################################################

# Pyhton script which plots the hystograms of the 103rd training brain

#################################################################

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from detect_peaks import detect_peaks

#import sklearn as sk
#from sklearn.linear_model import Lasso

import os, sys

#'''
X = 175
Y = 207
Z = 175
''' #for testing
X = 50
Y = 50
Z = 50
'''

#relative path
train103 = nib.load("../data/set_train/train_103.nii")

data = train103.get_data()

# store all the non zero values in a 1D list
intList=[]
for x in range(X):
    for y in range(Y):
        for z in range(Z):
			if data[x,y,z]!=0:
				intList.append(int(data[x,y,z]))

# compute (and plote if show=True) the peaks, and print them
histo = plt.figure(0)
values=plt.hist(intList, 200)
peakIndexes=detect_peaks(values[0], mph = 100, mpd = 30, show=True)
peaks=[]
print("There are "+str(len(peakIndexes))+" peaks:")
print(peakIndexes)
for i in range(len(peakIndexes)):
    peaks.append([values[1][peakIndexes[i]],values[0][peakIndexes[i]]])
    print("Peak "+str(i)+": ")
    print(peaks[i])
