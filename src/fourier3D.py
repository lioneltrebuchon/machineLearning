#####################################################################################################

# Python script which computes a real Fourier transform 

#####################################################################################################

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt

from detect_peaks import detect_peaks

import os, sys

#'''
X = 176
Y = 208
Z = 176
T = 100
''' #for testing
X = 50
Y = 50
Z = 50
T = 3
'''

train = [None]*T
data = [None]*T

ages = np.genfromtxt('../data/targets.csv', delimiter="\n")
peaksFile = open('../results/2peaks.csv','w')
colormap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['black','white'],256)

t = np.arange(start=1,stop=T,step=1) 

frequencies_mask_xy = np.ones([X,Y])
for i in xrange(20):
	for j in xrange(20):
		frequencies_mask_xy[ X*0.5 + i - 10 ][ Y*0.5 + j - 10 ] = 0.0
frequencies_mask_xz = np.ones([X,Z])
for i in xrange(20):
	for j in xrange(20):
		frequencies_mask_xz[ X*0.5 + i - 10 ][ Z*0.5 + j - 10 ] = 0.0
frequencies_mask_yz = np.ones([Y,Z])
for i in xrange(20):
	for j in xrange(20):
		frequencies_mask_yz[ Y*0.5 + i - 10 ][ Z*0.5 + j - 10 ] = 0.0

for i in t:
	print("Computing peaks of train"+str(i+1)+"...")
	peaksFile.write(str(i+1)+', ')
	peaksFile.write(str(ages[i])+', ') 
	#relative path
	train[i] = nib.load("../data/set_train/train_"+str(i+1)+".nii")

	data[i] = train[i].get_data()

	data_xy = np.squeeze(np.mean(data[i],axis=2),axis=2)
	data_xz = np.squeeze(np.mean(data[i],axis=1),axis=2)
	data_yz = np.squeeze(np.mean(data[i],axis=0),axis=2)

	data_xy = np.fft.fft2(data_xy)
	fouriers = plt.figure(0)
	data_xy = np.log(np.abs(np.fft.fftshift(data_xy))**2)
	data_xy = np.multiply(data_xy,frequencies_mask_xy)
	plt.imshow(data_xy)
	plt.title("Fourier in XY (age "+str(int(ages[i]))+")")
	fouriers.savefig("../plots/fourier_xy_age"+str(int(ages[i]))+"_train"+str(i)+".png")

	data_xz = np.fft.fft2(data_xz)
	fouriers = plt.figure(1)
	data_xz = np.log(np.abs(np.fft.fftshift(data_xz))**2)
	data_xz = np.multiply(data_xz,frequencies_mask_xz)
	plt.imshow(data_xz)
	plt.title("Fourier in XZ (age "+str(int(ages[i]))+")")
	fouriers.savefig("../plots/fourier_xz_age"+str(int(ages[i]))+"_train"+str(i)+".png")
	data_yz = np.fft.fft2(data_yz)
	fouriers = plt.figure(2)
	data_yz = np.log(np.abs(np.fft.fftshift(data_yz))**2)
	data_yz = np.multiply(data_yz,frequencies_mask_yz)
	plt.imshow(data_yz)
	plt.title("Fourier in YZ (age "+str(int(ages[i]))+")")
	fouriers.savefig("../plots/fourier_yz_age"+str(int(ages[i]))+"_train"+str(i)+".png")
	#plt.show()
	# plt.figure(1)
	# plt.imshow(data_xz,interpolation='nearest',
 #                    cmap=colormap,
 #                    origin='lower')
	# plt.show()


	# plt.figure(2)
	# plt.imshow(data_yz,interpolation='nearest',
 #                    cmap=colormap,
 #                    origin='lower')
	# plt.show()

