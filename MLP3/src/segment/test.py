###########################################################################################################################

# Python script which computes all the peaks features (p1_x, p1_y, p2_x, p2_y, p3_x p3_y) for the train set and save them in separated files

###########################################################################################################################

import numpy as np
import nibabel as nib

from detect_peaks import detect_peaks

#''' (remove/add # to switch)
X = 176
Y = 208
Z = 176
i = 1

train = nib.load("../../data/set_train/train_"+str(i)+".nii")
data = train.get_data()

hist = np.histogram(data,'auto',range = (10**(-10),2000))
valleyIndexes=detect_peaks(hist[0], mpd = len(hist[0])/4+10, valley=True, show=True)
peakIndexes=detect_peaks(hist[0], mph = 1000, mpd = len(hist[0])/4+10, valley=False, show=True)
if len(peakIndexes)==3: #the first peak doesn't exist each time
    for v in valleyIndexes:
        if v > peakIndexes[1] and v < peakIndexes[2]:
            threshold = hist[1][v]
            print("threshold = "+str(threshold)+"\n")
elif len(peakIndexes)==2: #the first peak doesn't exist each time
    for v in valleyIndexes:
        if v > peakIndexes[0] and v < peakIndexes[1]:
            threshold = hist[1][v]
            print("threshold = "+str(threshold)+"\n")
