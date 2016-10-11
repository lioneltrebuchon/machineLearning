########################################################################

# Pyhton script which plots the first brain (train1) projected onto the 3 plans

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
maxVal = 2900

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

#black->white colormap
colormap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['black','white'],256)


##### calculate color histograph over 1 picture

intList=np.empty(X*Y*Z)
for x in range(X):
    for y in range(Y):
        for z in range(Z):
			intList[z+(y+x*Y)*Z]=int(round(data[x,y,z]))


# plot and save figure
histo = plt.figure(0)
plt.hist(intList)
plt.title("Histogram of train1")
plt.xlabel("Values of the 3D brain")
plt.ylabel("Frequencies")
histo.savefig("../plots/plot_train1_histo.png")
