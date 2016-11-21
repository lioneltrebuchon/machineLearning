###########################################################################################################################

# Python script which computes all the peaks features (p1_x, p1_y, p2_x, p2_y, p3_x p3_y) for the train set and save them in separated files

###########################################################################################################################

import numpy as np
import nibabel as nib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys

from detect_peaks import detect_peaks

MPH = [5000, 6500, 8000, 9500, 1100]
MPD = [15,20,22,24,26,28,30]

#''' (remove/add # to switch)
X = 176
Y = 208
Z = 176
N_TRAIN = 278
''' #for testing with smaller values
X = 50
Y = 50
Z = 50
N_TRAIN = 2
#'''

train = [None]*N_TRAIN
data = [None]*N_TRAIN

files = []

for mph in MPH:
    files.append([open("../preproc/mph"+str(mph)+"_mpd"+str(mpd)+"_from"+sys.argv[1]+"to"+sys.argv[2], "w") for mpd in MPD])

for i in range(int(sys.argv[1])-1,int(sys.argv[2])):
    print("Computing features of train"+str(i+1)+"...")

    train[i] = nib.load("../data/set_train/train_"+str(i+1)+".nii")
    data[i] = train[i].get_data()

    # store all the non zero values in a 1D list
    intList=[]
    vol = 0
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if data[i][x,y,z]!=0:
                    intList.append(int(data[i][x,y,z]))
                    vol+=1

    # compute the peaks and save them
    values=plt.hist(intList, 200)
    for h in range(len(MPH)):
    	for d in range(len(MPD)):
    		peakIndexes=detect_peaks(values[0], MPH[h], MPD[d], show=False)
    		files[h][d].write(str(len(peakIndexes)))

for h in range(len(MPH)):
    for d in range(len(MPD)):
    	files[h][d].close()
