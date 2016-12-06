########################################################################

# Python script to perform segmentation

########################################################################

TEST=138
TRAIN=278
X = 176
Y = 208
Z = 176

z = 100 #which slice ?
threshold=1000 #default if detect peacts doesn't work


import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import svm
from scipy import ndimage
from detect_peaks import detect_peaks

image = nib.load("../data/set_train/train_17.nii")
data = image.get_data()
slice = data[:,:,:,0]
#slice = ndimage.gaussian_filter(slice, sigma=3)

list=[]
for x in range(X):
    for y in range(Y):
        for z in range(Z):
                if slice[x,y,z]!=0 and slice[x,y,z]<1500:
                    list.append(int(slice[x,y,z]))

hist=plt.hist(list, 200)
valleyIndexes=detect_peaks(hist[0], mpd = 30, valley=True, show=True)
peakIndexes=detect_peaks(hist[0], mph = 1000, mpd = 30, valley=False, show=True)
if len(peakIndexes)==3: #the first peak doesn't exist each time
    for v in valleyIndexes:
        if v > peakIndexes[1] and v < peakIndexes[2]:
            threshold = hist[1][v]
            print("threshold = "+str(threshold)+"\n")
elif len(peakIndexes)==2: #the first peak doesn't exist each time
    for v in valleyIndexes:
        if v > peakIndexes[0] and v < peakIndexes[1]:
            threshold = hist[1][v]
            print("threshold = "+str(threshold)+"\n")

whiteMask = (slice > threshold)

surface = np.sum(whiteMask)
print("surface = "+str(surface)+"\n")

perimeter = np.sum(whiteMask[:,1:] != whiteMask[:,:-1]) + np.sum(whiteMask[1:,:] != whiteMask[:-1,:])
print("perimeter = "+str(perimeter)+"\n")

whiteMatter = (slice > threshold)*slice
grayMatter = (slice < threshold)*(slice > 0)*slice

bothMatters = (slice > threshold)*slice+(slice < threshold)*(slice > 0)*(-slice)

colormap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['black','white'],256)
doublemap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['red','black','blue'],256)

norm = plt.Normalize(0, 2000)
doublenorm = plt.Normalize(-2000, 2000)

plt.figure(10)
plt.imshow(slice, cmap = colormap, norm=norm)
plt.figure(20)
plt.imshow(whiteMask, cmap = colormap)
plt.figure(30)
plt.imshow(bothMatters, cmap = doublemap, norm=doublenorm)
plt.show()
plt.close()
