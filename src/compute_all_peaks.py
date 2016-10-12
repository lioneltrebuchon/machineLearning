########################################################################

# Python script which plots an histogram for each training brain

########################################################################

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

train = [None]*T
data = [None]*T

ages = np.genfromtxt('../data/targets.csv', delimiter="\n")
peaksFile = open('../results/peaks.csv','w')

for i in range(T):
    print("Computing peaks of train"+str(i+1)+"...")
    peaksFile.write(str(i+1)+', ') 
    #relative path
    train[i] = nib.load("../data/set_train/train_"+str(i+1)+".nii")

    data[i] = train[i].get_data()

    # store all the non zero values in a 1D list
    intList=[]
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if data[i][x,y,z]!=0:
                    intList.append(int(data[i][x,y,z]))

    # compute the peaks, print them, and save them in results/peaks.csv
    values=plt.hist(intList, 200)
    peakIndexes=detect_peaks(values[0], mph = 100, mpd = 30, show=False)
    peaks=[]
    print("There are "+str(len(peakIndexes))+" peaks:")
    peaksFile.write(str(len(peakIndexes))) 
    print(peakIndexes)
    for i in range(len(peakIndexes)):
        peaks.append([values[1][peakIndexes[i]],values[0][peakIndexes[i]]])
        print("Peak "+str(i)+": ")
        print(peaks[i])
        peaksFile.write(", "+str(peaks[i]))
    peaksFile.write("\n")
peaksFile.close()
