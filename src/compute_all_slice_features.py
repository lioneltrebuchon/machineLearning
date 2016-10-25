###########################################################################################################################

# Python script which computes all the features (vol, p1_x, p1_y, p2_x, p2_y, p3_x p3_y) and save them in separated files

###########################################################################################################################

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt

from detect_peaks import detect_peaks

import os, sys

#'''
X = 176
Y = 208
Z = 176
T = 278
''' #for testing
X = 50
Y = 50
Z = 50
T = 3
'''

Ncuts = 16
Nranges = 16


train = [None]*T
data = [None]*T

ages = np.genfromtxt('../data/targets.csv', delimiter="\n")

featureFiles = {}

for a in ['x','y','z']:
    for c in range(Ncuts):
        for r in range(Nranges):
            acr = a+format(c, 'x')+format(r, 'x')
            featureFiles[acr] = open('../results/sliceFeatures/'+acr+'.csv','w')
MAX=0
for i in range(T):
    print("Computing features of train"+str(i+1)+"...")
    
    train[i] = nib.load("../data/set_train/train_"+str(i+1)+".nii")
    data[i] = train[i].get_data()
    
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                intensity = data[i][x,y,z]
                if intensity > MAX:
                    MAX = intensity

for a in ['x','y','z']:
    for c in range(Ncuts):
        for r in range(Nrange):
            acr = a+format(str(c), '02x')+format(str(r), '02x')
            featureFiles[acr].write(MAX)
            featureFiles[acr].close()
