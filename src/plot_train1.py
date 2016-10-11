########################################################################

# Pyhton script which plots the first brain (train1) projected onto the 3 plans

########################################################################

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt

#import sklearn as sk
#from sklearn.linear_model import Lasso

import os, sys

#'''
X = 175
Y = 207
Z = 175

''' #for testing
X = 10
Y = 10
Z = 10
'''

#relative path
train1 = nib.load("../data/set_train/train_1.nii")

#path from usb key
#train1 = nib.load("/run/media/lionelt/04F6-B693/ML/data/set_train/train_1.nii")
data = train1.get_data()

#black->white colormap
colormap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['black','white'],256)

##### plot the 1st brain projected on x and y axis

#compute means along z-axis
mean_z=np.empty([X,Y])
for x in range(X):
    for y in range(Y):
	mean_z[x,y] = 0
        for z in range(Z):
            	mean_z[x,y]+=data[x, y, z]
	mean_z[x,y]/=Z

#plot and save figure
fig_xy = plt.figure(0)
img_xy = plt.imshow(mean_z,interpolation='nearest',
                    cmap=colormap,
                    origin='lower')
plt.colorbar(img_xy,cmap=colormap)
fig_xy.savefig("plots/plot_train1_xy.png")

##### plot the 1st brain projected on x and z axis

#compute means along y-axis
mean_y=np.empty([X,Z])
for x in range(X):
    for z in range(Z):
	mean_y[x,z] = 0
        for y in range(Y):
		mean_y[x,z]+=data[x, y, z]
	mean_y[x,z]/=Y

#plot and save figure
fig_xz = plt.figure(1)
img_xz = plt.imshow(mean_y,interpolation='nearest',
                    cmap=colormap,
                    origin='lower')
plt.colorbar(img_xz,cmap=colormap)
fig_xz.savefig("plots/plot_train1_xz.png")

##### plot the 1st brain projected on z and y axis

#compute means along x-axis
mean_x=np.empty([Z,Y])
for z in range(Z):
    for y in range(Y):
	mean_x[z,y] = 0
        for x in range(Z):
        	mean_x[z,y]+=data[x, y, z]
	mean_x[z,y]/=X

#plot and save figure
fig_zy = plt.figure(2)
img_zy = plt.imshow(mean_x,interpolation='nearest',
                    cmap=colormap,
                    origin='lower')
plt.colorbar(img_zy,cmap=colormap)
fig_zy.savefig("plots/plot_train1_zy.png")
