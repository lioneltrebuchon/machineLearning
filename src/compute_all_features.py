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

train = [None]*T
data = [None]*T

ages = np.genfromtxt('../data/targets.csv', delimiter="\n")
fvol = open('../results/vol.csv','w')
fp1_x = open('../results/p1_x.csv','w')
fp1_y = open('../results/p1_y.csv','w')
fp2_x = open('../results/p2_x.csv','w')
fp2_y = open('../results/p2_y.csv','w')
fp3_x = open('../results/p3_x.csv','w')
fp3_y = open('../results/p3_y.csv','w')

for i in range(T):
    print("Computing features of train"+str(i+1)+"...")
    
    train[i] = nib.load("../data/set_train/train_"+str(i+1)+".nii")
    data[i] = train[i].get_data()
    
    # store all the non zero values in a 1D list and compute the volume
    intList=[]
    vol = 0
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if data[i][x,y,z]!=0:
                    intList.append(int(data[i][x,y,z]))
                    vol+=1
                    
    # save the volume                
    fvol.write(str(vol)+"\n")
    
    # compute the peaks and save them
    values=plt.hist(intList, 200)
    peakIndexes=detect_peaks(values[0], mph = 1000, mpd = 30, show=False)
    peaks=[]
    if len(peakIndexes)==2:
        fp1_x.write("0\n")
        fp1_y.write("0\n")
        fp2_x.write(str(values[1][peakIndexes[0]])+"\n")
        fp2_y.write(str(values[0][peakIndexes[0]])+"\n")
        fp3_x.write(str(values[1][peakIndexes[1]])+"\n")
        fp3_y.write(str(values[0][peakIndexes[1]])+"\n")
    elif len(peakIndexes)==3:
        fp1_x.write(str(values[1][peakIndexes[0]])+"\n")
        fp1_y.write(str(values[0][peakIndexes[0]])+"\n")
        fp2_x.write(str(values[1][peakIndexes[1]])+"\n")
        fp2_y.write(str(values[0][peakIndexes[1]])+"\n")
        fp3_x.write(str(values[1][peakIndexes[2]])+"\n")
        fp3_y.write(str(values[0][peakIndexes[2]])+"\n")
    else:
        print("Error: number of peaks different from 2 or 3 for the train "+str(i+1))
        fp1_x.write("0\n")
        fp1_y.write("0\n")
        fp2_x.write("0\n")
        fp2_y.write("0\n")
        fp3_x.write("0\n")
        fp3_y.write("0\n")
        
fvol.close()
fp1_x.close()
fp1_y.close()
fp2_x.close()
fp2_y.close()
fp3_x.close()
fp3_y.close()
