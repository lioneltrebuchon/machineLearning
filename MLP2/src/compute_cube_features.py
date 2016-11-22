###########################################################################################################################

# Python script which computes all the section features of the train/test set and save them in 1 .csv file

###########################################################################################################################

import numpy as np
import nibabel as nib
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

N_SECTIONS = 8 #8 sections in each direction, so 8^3 sections in total
N_MARGIN = 2 # cubes that we don't take
N_RANGES = 48   #48 ranges of intensity for the histogram
''' #for training with smaller values
X = 50
Y = 50
Z = 50
N_IMAGES = 2
N_SECTIONS = 5
N_RANGES = 5
#'''

maxIntensity = 2000  # test whether this value should be like this or even lower!
sizeRange = maxIntensity / N_RANGES

sizeSectionX = X/N_SECTIONS
sizeSectionY = Y/N_SECTIONS
sizeSectionZ = Z/N_SECTIONS
N_CUBES = (N_SECTIONS-2*N_MARGIN)**3

images = [None]*N_IMAGES
data = [None]*N_IMAGES

featuresFile = open('../features/'+imagesSet+'_cube_features_from'+firstImage+'to'+lastImage+'.csv','w')

for i in xrange(int(firstImage)-1,min(int(lastImage), N_IMAGES):
    print("Computing features of "+imagesSet+str(i+1)+"...")

    images[i] = nib.load("../data/set_"+imagesSet+"/"+imagesSet+"_"+str(i+1)+".nii")
    data[i] = images[i].get_data()

    feature = np.zeros((3,N_SECTIONS,N_RANGES),np.uint)
    cube = np.zeros( (N_CUBES, N_RANGES),np.uint)
    flag = np.zeros(N_CUBES,np.uint)

    # Compute the features: each pixel is exactly in 3 sections (an x section, an y section, and a z section) so we go through all the pixels and change the values of the 3 corresponding features

    for x in xrange(X):
        secX = x/sizeSectionX
        if secX >= N_MARGIN and secX < N_SECTIONS-N_MARGIN:
            for y in xrange(Y):
                secY = y/sizeSectionY
                if secY >= N_MARGIN and secY < N_SECTIONS-N_MARGIN:
                    for z in xrange(Z):
                        secZ = z/sizeSectionZ
                        if secZ >= N_MARGIN and secZ < N_SECTIONS-N_MARGIN:
                            D = data[i][x,y,z]
                            cubeNr = (secX-N_MARGIN) + (secY-N_MARGIN)*(N_SECTIONS-2*N_MARGIN) + (secZ-N_MARGIN)*(N_SECTIONS-2*N_MARGIN)**2
                            if D>0:
                                r = D/sizeRange
                                if r<N_RANGES:
            						cube[ cubeNr , r] += 1
                                    # feature[0,secX,r]+=1
                                    # feature[1,secY,r]+=1
                                    # feature[2,secZ,r]+=1

    # Write the features in the .csv file: 1 line per image


    for s in xrange(N_CUBES):
        for r in xrange(N_RANGES):
            if s+r != 0:
                featuresFile.write(',')
            featuresFile.write(str(cube[s,r]))
    featuresFile.write('\n')

featuresFile.close()
