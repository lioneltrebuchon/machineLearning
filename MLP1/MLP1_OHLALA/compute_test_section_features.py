###########################################################################################################################

# Python script which computes all the section features of the test set and save them in 1 .csv file

###########################################################################################################################

import numpy as np
import nibabel as nib

#''' (remove/add # to switch)
X = 176
Y = 208
Z = 176
N_TEST = 138
N_SECTIONS = 16 #16 sections in each direction, so 48 sections in total
N_RANGES = 48   #48 ranges of intensity for the histogram
''' #for testing with smaller values
X = 50
Y = 50
Z = 50
N_TEST = 2
N_SECTIONS = 5
N_RANGES = 5
#'''

maxIntensity = 2400
sizeRange = maxIntensity / N_RANGES

sizeSectionX = X/N_SECTIONS
sizeSectionY = Y/N_SECTIONS
sizeSectionZ = Z/N_SECTIONS

test = [None]*N_TEST
data = [None]*N_TEST

featuresFile = open('features/test_section_features.csv','w')

for i in xrange(N_TEST):
    print("Computing features of test"+str(i+1)+"...")
    
    test[i] = nib.load("data/set_test/test_"+str(i+1)+".nii")
    data[i] = test[i].get_data()
    
    feature = np.zeros((3,N_SECTIONS,N_RANGES),np.uint)    

    # Compute the features: each pixel is exactly in 3 sections (an x section, an y section, and a z section) so we go through all the pixels and change the values of the 3 corresponding features

    for x in xrange(X):
        secX = x/sizeSectionX
        for y in xrange(Y):
            secY = y/sizeSectionY
            for z in xrange(Z):
                secZ = z/sizeSectionZ
                D = data[i][x,y,z]
                if D>0:
                    r = D/sizeRange
                    if r<16:
                        feature[0,secX,r]+=1
                        feature[1,secY,r]+=1
                        feature[2,secZ,r]+=1

    # Write the features in the .csv file: 1 line per image

    for a in xrange(3):
        for s in xrange(N_SECTIONS):
            for r in xrange(N_RANGES):
                if a+s+r != 0:
                    featuresFile.write(',')
                featuresFile.write(str(feature[a,s,r]))
    
    featuresFile.write('\n')
    
featuresFile.close()
