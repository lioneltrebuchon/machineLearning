###########################################################################################################################

# Python script which computes all the section features of the train set and save them in 1 .csv file

###########################################################################################################################

import numpy as np
import nibabel as nib

#''' (remove/add # to switch)
X = 176
Y = 208
Z = 176
if sys.argv[3] == 'train':
    N_TRAIN = 278
elif sys.argv[3] == 'test':
    N_TRAIN = 1
N_SECTIONS = 16 #16 sections in each direction, so 3*16=48 sections in total
N_RANGES = 48   #48 ranges of intensity for the histogram
''' #for training with smaller values
X = 50
Y = 50
Z = 50
N_TRAIN = 2
N_SECTIONS = 5
N_RANGES = 5
#'''

maxIntensity = 2400
sizeRange = maxIntensity / N_RANGES

sizeSectionX = X/N_SECTIONS
sizeSectionY = Y/N_SECTIONS
sizeSectionZ = Z/N_SECTIONS

train = [None]*N_TRAIN
data = [None]*N_TRAIN

featuresFile = open('../features/'+sys.argv[3]+'_section_features'+sys.arg[1]+'to'+sys.argv[2]+'.csv','w')

for i in xrange(int(sys.argv[1]-1),int(sys.argv[2]):
    print("Computing features of train"+str(i+1)+"...")
    
    train[i] = nib.load("../data/set_"+sys.argv[3]+"/"+sys.argv[3]+"_"+str(i+1)+".nii")
    data[i] = train[i].get_data()
    
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
                    if r<N_RANGES:
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
