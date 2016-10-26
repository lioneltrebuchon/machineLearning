###########################################################################################################################

# Python script which computes all the slice features and save them in 1 file

###########################################################################################################################

import numpy as np
import nibabel as nib

#'''
X = 176
Y = 208
Z = 176
T = 278
Ncuts = 16
Nranges = 16
''' #for testing
X = 50
Y = 50
Z = 50
T = 2
Ncuts = 5
Nranges = 5
#'''



maxIntensity = 2400
sizeRange = maxIntensity / Nranges

sizeCutX = X/Ncuts
sizeCutY = Y/Ncuts
sizeCutZ = Z/Ncuts

axes = ['x','y','z']

train = [None]*T
data = [None]*T

''' Separate files
featureFiles = {}

for a in xrange(3):
    for c in xrange(Ncuts):
        for r in xrange(Nranges):
            acr = axes[a]+format(c, 'x')+format(r, 'x')
            featureFiles[acr] = open('../results/sliceFeatures/'+acr+'.csv','w')
'''

featuresFile = open('../results/sliceFeatures2.csv','w')

for i in xrange(108,192):
    print("Computing features of train"+str(i+1)+"...")
    
    train[i] = nib.load("../data/set_train/train_"+str(i+1)+".nii")
    data[i] = train[i].get_data()
    
    feature = np.zeros((3,Ncuts,Nranges),np.uint)    

    for x in xrange(X):
        cx = x/sizeCutX
        for y in xrange(Y):
            cy = y/sizeCutY
            for z in xrange(Z):
                cz = z/sizeCutZ
                D = data[i][x,y,z]
                if D>0:
                    r = D/sizeRange
                    if r<16:
                        feature[0,cx,r]+=1
                        feature[1,cy,r]+=1
                        feature[2,cz,r]+=1
    ''' Separate files                
    for a in xrange(3):
        for c in xrange(Ncuts):
            for r in xrange(Nrange):
                acr = axes[a]+format(str(c), '02x')+format(str(r), '02x')
                featureFiles[acr].write(str(feature[a,c,r])+'\n'
    '''
                
    for a in xrange(3):
        for c in xrange(Ncuts):
            for r in xrange(Nranges):
                if a+c+r != 0:
                    featuresFile.write(',')
                featuresFile.write(str(feature[a,c,r]))
    
    featuresFile.write('\n')
    
''' Separate files
for a in xrange(3):
    for c in xrange(Ncuts):
        for r in xrange(Nrange):
            acr = axes[a]+format(str(c), '02x')+format(str(r), '02x')
            featureFiles[acr].close()
'''
featuresFile.close()
