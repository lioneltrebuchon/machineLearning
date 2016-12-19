###########################################################################################################################

# Python script which computes all the segment features of the train/test set and save them in 1 .csv file

###########################################################################################################################

import numpy as np
import nibabel as nib
'''import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt'''
from detect_peaks import detect_peaks
import sys

#''' (remove/add # to switch)
X = 176
Y = 208
Z = 176

N_SECTIONS = 16
N_MARGIN = 3
''' #for training with smaller values
X = 50
Y = 50
Z = 50
N_SECTIONS = 5
#'''

imagesSet = "train"
if len(sys.argv)>1:
    imagesSet = sys.argv[1]
if imagesSet == "test":
    N_IMAGES = 138
else:
    N_IMAGES = 278
firstImage = '1'
if len(sys.argv)>2:
    firstImage = sys.argv[2]
lastImage = str(N_IMAGES)
if len(sys.argv)>3:
    lastImage = sys.argv[3]

sizeSectionX = X/N_SECTIONS
sizeSectionY = Y/N_SECTIONS
sizeSectionZ = Z/N_SECTIONS

images = [None]*N_IMAGES
data = [None]*N_IMAGES

featuresFile = open('../features/segment/'+imagesSet+'_segment32_features_from'+firstImage+'to'+lastImage+'.csv','w')

for i in xrange(int(firstImage)-1,min(int(lastImage), N_IMAGES)):
    print("\n\nComputing features of "+imagesSet+str(i+1)+"...")

    images[i] = nib.load("../data/set_"+imagesSet+"/"+imagesSet+"_"+str(i+1)+".nii")
    data[i] = images[i].get_data()

    features = np.zeros((2,3,N_SECTIONS-2*N_MARGIN),np.uint)

    # Compute the features: each pixel is exactly in 3 sections (an x section, an y section, and a z section) so we go through all the pixels and change the values of the 3 corresponding features
    for d in xrange(3):
        for s in xrange(N_MARGIN,N_SECTIONS-N_MARGIN):
            print("\nSlice "+['X','Y','Z'][d]+str(s))
            if d==0:
                slice = data[i][s*sizeSectionX:(s+1)*sizeSectionX,:,:,0]
            elif d==1:
                slice = data[i][:,s*sizeSectionY:(s+1)*sizeSectionY,:,0]
            else:
                slice = data[i][:,:,s*sizeSectionZ:(s+1)*sizeSectionZ,0]
            threshold = 1000
            hist = np.histogram(slice,'auto',range = (10**(-10),2000))
            valleyIndexes=detect_peaks(hist[0], mpd = len(hist[0])/4, valley=True, show=False)
            peakIndexes=detect_peaks(hist[0], mph = 1000, mpd = len(hist[0])/5-5, valley=False, show=False)
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
            print("threshold = "+str(threshold))

            whiteMask = (slice > threshold)

            volume = np.sum(whiteMask)
            print("volume = "+str(volume))
            features[0,d,s-N_MARGIN]=volume

            frontier = np.sum(whiteMask[:,:,1:] != whiteMask[:,:,:-1]) + np.sum(whiteMask[:,1:,:] != whiteMask[:,:-1,:]) + np.sum(whiteMask[:,:,1:] != whiteMask[:,:,:-1])
            print("frontier = "+str(frontier))
            features[1,d,s-N_MARGIN]=frontier

    # Write the features in the .csv file: 1 line per image
    for f in xrange(2):
        for d in xrange(3):
            for s in xrange(N_SECTIONS-2*N_MARGIN):
                if f+d+s!=0:
                    featuresFile.write(',')
                featuresFile.write(str(features[f,d,s]))
    featuresFile.write('\n')

featuresFile.close()
