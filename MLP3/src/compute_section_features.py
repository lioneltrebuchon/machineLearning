###########################################################################################################################

# Python script which computes all the section features of the train/test set and save them in 1 .csv file

###########################################################################################################################

import numpy as np
import nibabel as nib
import sys

#''' (remove/add # to switch)
X = 176
Y = 208
Z = 176

N_BINS = 48
N_SECTIONS = 16
N_MARGIN = 0
MAX = 2400

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

#featuresFile = open('../features/section/'+imagesSet+'_section_features_from'+firstImage+'to'+lastImage+'.csv','w')
featuresFile = open('../features/'+imagesSet+'_section_features_bis.csv','w')

for i in xrange(int(firstImage)-1,min(int(lastImage), N_IMAGES)):
    print("\n\nComputing features of "+imagesSet+str(i+1)+"...")

    image = nib.load("../data/set_"+imagesSet+"/"+imagesSet+"_"+str(i+1)+".nii")
    data = image.get_data()

    features = np.zeros((2,3,N_SECTIONS-2*N_MARGIN),np.uint)

    # Compute the features: each pixel is exactly in 3 sections (an x section, an y section, and a z section) so we go through all the pixels and change the values of the 3 corresponding features
    for d in xrange(3):
        for s in xrange(N_MARGIN,N_SECTIONS-N_MARGIN):
            #print("\nSlice "+['X','Y','Z'][d]+str(s))
            if d==0:
                slice = data[s*sizeSectionX:(s+1)*sizeSectionX,:,:,0]
            elif d==1:
                slice = data[:,s*sizeSectionY:(s+1)*sizeSectionY,:,0]
            else:
                slice = data[:,:,s*sizeSectionZ:(s+1)*sizeSectionZ,0]
            threshold = 1000
            hist = np.histogram(slice,N_BINS,range = (10**(-10),MAX))
            n_bins=len(hist[0])
            for b in xrange(n_bins):
                if d+s+b!=0:
                    featuresFile.write(',')
                featuresFile.write(str(hist[0][b]))
    featuresFile.write('\n')

featuresFile.close()
