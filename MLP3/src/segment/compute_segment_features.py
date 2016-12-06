###########################################################################################################################

# Python script which computes all the section features of the train/test set and save them in 1 .csv file

###########################################################################################################################

import numpy as np
import nibabel as nib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from detect_peaks import detect_peaks
import sys

imagesSet = sys.argv[1]
firstImage = sys.argv[2]
lastImage = sys.argv[3]

#''' (remove/add # to switch)
X = 176
Y = 208
Z = 176

if imagesSet == "test":
    N_IMAGES = 138
elif imagesSet == "train":
    N_IMAGES = 278
else:
    print("You should specify 'train' or 'test'")

N_SECTIONS = 16
''' #for training with smaller values
X = 50
Y = 50
Z = 50
N_SECTIONS = 5
#'''

sizeSectionX = X/N_SECTIONS
sizeSectionY = Y/N_SECTIONS
sizeSectionZ = Z/N_SECTIONS

images = [None]*N_IMAGES
data = [None]*N_IMAGES

featuresFile = open('../features/segment/'+imagesSet+'_segment_features_from'+firstImage+'to'+lastImage+'.csv','w')

for i in xrange(int(firstImage)-1,min(int(lastImage), N_IMAGES)):
    print("Computing features of "+imagesSet+str(i+1)+"...")

    images[i] = nib.load("../data/set_"+imagesSet+"/"+imagesSet+"_"+str(i+1)+".nii")
    data[i] = images[i].get_data()

    feature = np.zeros((3,2,N_SECTIONS),np.uint)

    # Compute the features: each pixel is exactly in 3 sections (an x section, an y section, and a z section) so we go through all the pixels and change the values of the 3 corresponding features

    for secX in xrange(N_SECTIONS):
        slice = data[secX*sizeSectionX:(secX+1)*sizeSectionX,:,:,0]
        list=[]
        for x in range(X):
            for y in range(Y):
                for y in range(Y):############ J'en suis là
                        if slice[x,y]!=0:
                            list.append(int(slice[x,y]))
        hist=plt.hist(list, 100)
        peakIndexes=detect_peaks(hist[0], mpd = 30, valley=True, show=False)
        if len(peakIndexes)==3: #the first peak doesn't exist each time
            threshold = hist[1][peakIndexes[1]]
        else:
            print("Error: number of valleys different from 3\n")
        print("threshold = "+str(threshold)+"\n")


        whiteMask = (slice > threshold)

        surface = np.sum(whiteMask)
        print("surface = "+str(surface)+"\n")

        perimeter = np.sum(whiteMask[:,1:] != whiteMask[:,:-1]) + np.sum(whiteMask[1:,:] != whiteMask[:-1,:])
        print("perimeter = "+str(perimeter)+"\n")

    # Write the features in the .csv file: 1 line per image


    for s in xrange(N_CUBES):
        for r in xrange(N_RANGES):
            if s+r != 0:
                featuresFile.write(',')
            featuresFile.write(str(cube[s,r]))
    featuresFile.write('\n')

featuresFile.close()
