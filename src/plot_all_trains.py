########################################################################

# Python script which plots all the brains projected onto the 3 plans

########################################################################

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt

#import sklearn as sk
#from sklearn.linear_model import Lasso

import os, sys

#'''
X = 176
Y = 208
Z = 176
T = 278
''' #for testing
X = 10
Y = 10
Z = 10
T = 2
'''

train = [None]*T
data = [None]*T

ages = np.genfromtxt('../data/targets.csv', delimiter="\n")

#black->white colormap
colormap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['black','white'],256)

for i in range(T):
	print("plotting train"+str(i+1)+"...")
	#relative path
	train[i] = nib.load("../data/set_train/train_"+str(i+1)+".nii")

	data[i] = train[i].get_data()

	print("computing XY...")
	#compute means along z-axis
	mean_z=np.empty([X,Y])
	for x in range(X):
	    for y in range(Y):
		mean_z[x,y] = 0
		for z in range(Z):
		    	mean_z[x,y]+=data[i][x, y, z]
		mean_z[x,y]/=Z

	print("computing XZ...")
	#compute means along y-axis
	mean_y=np.empty([X,Z])
	for x in range(X):
	    for z in range(Z):
		mean_y[x,z] = 0
		for y in range(Y):
			mean_y[x,z]+=data[i][x, y, z]
		mean_y[x,z]/=Y

	print("computing ZY...")
	#compute means along x-axis
	mean_x=np.empty([Z,Y])
	for z in range(Z):
	    for y in range(Y):
		mean_x[z,y] = 0
		for x in range(Z):
			mean_x[z,y]+=data[i][x, y, z]
		mean_x[z,y]/=X

	#plot the figure with 3 subplots
	fig = plt.figure(num=1, figsize=(16, 5))

	plt.subplot(131) # 1 row, 3 columns, 1st position
	img_xy = plt.imshow(mean_z,interpolation='nearest',
		            cmap=colormap,
		            origin='lower')
	#plt.colorbar(img_xy,cmap=colormap)
	plt.title("Train"+str(i+1)+" projected on XY (age "+str(int(ages[i]))+")")
	plt.xlabel("x")
	plt.ylabel("y")
	
	plt.subplot(132) # 1 row, 3 columns, 2nd position
	img_xz = plt.imshow(mean_y,interpolation='nearest',
		            cmap=colormap,
		            origin='lower')
	#plt.colorbar(img_xz,cmap=colormap)
	plt.title("Train"+str(i+1)+" projected on XZ (age "+str(int(ages[i]))+")")
	plt.xlabel("x")
	plt.ylabel("z")

	plt.subplot(133) # 1 row, 3 columns, 3rd position
	img_zy = plt.imshow(mean_x,interpolation='nearest',
		            cmap=colormap,
		            origin='lower')
	#plt.colorbar(img_zy,cmap=colormap)
	plt.title("Train"+str(i+1)+" projected on ZY (age "+str(int(ages[i]))+")")
	plt.xlabel("z")
	plt.ylabel("y")

	fig.savefig("../plots/plot_age"+str(int(ages[i]))+"_train"+str(i+1)+".png")
	plt.clf() #close figure
