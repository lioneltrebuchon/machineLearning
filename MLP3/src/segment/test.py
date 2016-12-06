###########################################################################################################################

# Python script which computes all the peaks features (p1_x, p1_y, p2_x, p2_y, p3_x p3_y) for the train set and save them in separated files

###########################################################################################################################

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt

from detect_peaks import detect_peaks

#''' (remove/add # to switch)
X = 176
Y = 208
Z = 176
i = 118
print("Computing features of train"+str(i)+"...")

train = nib.load("../data/set_train/train_"+str(i)+".nii")
data = train.get_data()

# store all the non zero values in a 1D list
intList=[]
vol = 0
for x in range(X):
    for y in range(Y):
        for z in range(Z):
            if data[x,y,z]!=0 and data[x,y,z]<1500:
                intList.append(int(data[x,y,z]))
                vol+=1

# compute the peaks and save them
hist=plt.hist(intList, 400)
valleyIndexes=detect_peaks(hist[0], mpd = 60, valley=True, show=True)
peakIndexes=detect_peaks(hist[0], mph = 1000, mpd = 120, valley=False, show=True)
minV = 20000
if len(peakIndexes)==3: #the first peak doesn't exist each time
    for v in valleyIndexes:
        if v > peakIndexes[1] and v < peakIndexes[2] and hist[0][v] < minV:
            threshold = hist[1][v]
            minV = hist[0][v]
elif len(peakIndexes)==2: #the first peak doesn't exist each time
    for v in valleyIndexes:
        if v > peakIndexes[0] and v < peakIndexes[1] and hist[0][v] < minV:
            threshold = hist[1][v]
            minV = hist[0][v]
print("threshold = "+str(threshold)+"\n")
