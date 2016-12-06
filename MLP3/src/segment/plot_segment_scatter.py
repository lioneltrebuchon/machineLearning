########################################################################

# Python script to perform segmentation

########################################################################

#''' (remove/add # to switch)
TRAIN=278
X = 176
Y = 208
Z = 176
''' #for training with smaller values
TRAIN=10
X = 176
Y = 208
Z = 176
#'''

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import svm
from scipy import ndimage
from detect_peaks import detect_peaks

images = [None]*TRAIN
data = [None]*TRAIN
genders = np.genfromtxt('correct_genders.csv', delimiter="\n")
colors1 = ['blue' if g==0 else 'red' for g in genders[0:TRAIN]]
ages = np.genfromtxt('correct_ages.csv', delimiter="\n")
healths = np.genfromtxt('correct_healths.csv', delimiter="\n")
ages_healths = 1+np.add(ages,healths) # = 1 if old sick, 2 if old healthy, 3 if young (because young=>healthy)
areas = [np.pi*(3*a_h)**2 for a_h in ages_healths[0:TRAIN]]
colors2 = ['green' if a_h==3 else 'red' for a_h in ages_healths]
volumes = np.genfromtxt('correct_volumes.csv', delimiter="\n")
frontiers = np.genfromtxt('correct_frontiers.csv', delimiter="\n")
'''
for i in xrange(249,TRAIN):
    print("Computing features of train "+str(i+1)+"...\n")
    images[i] = nib.load("../data/set_train/train_"+str(i+1)+".nii")
    data[i] = images[i].get_data()
    #slice = ndimage.gaussian_filter(slice, sigma=3)

    list=[]
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if data[i][x,y,z]!=0 and data[i][x,y,z]<1500:
                    list.append(int(data[i][x,y,z]))
    default=True
    hist=plt.hist(list, 100)
    valleyIndexes=detect_peaks(hist[0], mpd = 30, valley=True, show=False)
    peakIndexes=detect_peaks(hist[0], mpd = 30, valley=False, show=False)
    if len(peakIndexes)==3: #the first peak doesn't exist each time
        for v in valleyIndexes:
            if v > peakIndexes[1] and v < peakIndexes[2]:
                threshold = hist[1][v]
                default=False
    elif len(peakIndexes)==2: #the first peak doesn't exist each time
        for v in valleyIndexes:
            if v > peakIndexes[0] and v < peakIndexes[1]:
                threshold = hist[1][v]
                default=False
    if default:
        threshold=1000
        print("default ")
    print("threshold = "+str(threshold)+"\n")

    whiteMask = (data[i] > threshold)

    volume = np.sum(whiteMask)
    print("volume = "+str(volume)+"\n")
    volumes.append(volume)

    frontier = np.sum(whiteMask[:,:,1:] != whiteMask[:,:,:-1]) + np.sum(whiteMask[1:,:,:] != whiteMask[:-1,:,:]) + np.sum(whiteMask[:,1:,:] != whiteMask[:,:-1,:])
    print("frontier = "+str(frontier)+"\n")
    frontiers.append(frontier)
'''
plt.figure(1)
plt.scatter(volumes, frontiers, c=colors1, alpha=0.5)
plt.show()

plt.figure(2)
plt.scatter(volumes, frontiers, c=colors2, alpha=0.5)
plt.show()
