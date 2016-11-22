###########################################################################################################################

# Python script which computes all the section features of the train set and save them in 1 .csv file

###########################################################################################################################

import numpy as np
import nibabel as nib

#''' (remove/add # to switch)
X = 176
Y = 208
Z = 176
N_TRAIN = 278
N_SECTIONS = 8 #8 sections in each direction, so 8^3 sections in total
N_RANGES = 48   #48 ranges of intensity for the histogram
''' #for training with smaller values
X = 50
Y = 50
Z = 50
N_TRAIN = 2
N_SECTIONS = 5
N_RANGES = 5
#'''

maxIntensity = 2000  # test whether this value should be like this or even lower!
sizeRange = maxIntensity / N_RANGES

sizeSectionX = X/N_SECTIONS
sizeSectionY = Y/N_SECTIONS
sizeSectionZ = Z/N_SECTIONS
N_CUBES = N_SECTIONS*N_SECTIONS*N_SECTIONS

train = [None]*N_TRAIN
data = [None]*N_TRAIN

featuresFile = open('../features/train_cube_features.csv','w')

for i in xrange(N_TRAIN):
    print("Computing features of train"+str(i+1)+"...")

    train[i] = nib.load("/local/set_train/train_"+str(i+1)+".nii")
    data[i] = train[i].get_data()

    feature = np.zeros((3,N_SECTIONS,N_RANGES),np.uint)
    cube = np.zeros( (N_CUBES, N_RANGES),np.uint)
    flag = np.zeros(N_CUBES,np.uint) 

    # Compute the features: each pixel is exactly in 3 sections (an x section, an y section, and a z section) so we go through all the pixels and change the values of the 3 corresponding features

    for x in xrange(X):
        secX = x/sizeSectionX
        for y in xrange(Y):
            secY = y/sizeSectionY
            for z in xrange(Z):
                secZ = z/sizeSectionZ
                D = data[i][x,y,z]
                cubeNr = secX + secY*N_SECTIONS + secZ*N_SECTIONS*N_SECTIONS
                if D>0:
                    r = D/sizeRange
                    if r<N_RANGES:
						cube[ cubeNr , r] += 1
                        # feature[0,secX,r]+=1
                        # feature[1,secY,r]+=1
                        # feature[2,secZ,r]+=1
                else: # remember if a cube has 0 values inside, it is part of the outer brain
                    flag[cubeNr]=1

    # Write the features in the .csv file: 1 line per image


    for s in xrange(N_CUBES):
        for r in xrange(N_RANGES):
            if s+r != 0:
                featuresFile.write(',')
            # featuresFile.write(str(feature[s,r]))
            featuresFile.write(str(cube[s,r]))

	featuresFile.write('\n')

featuresFile.close()
