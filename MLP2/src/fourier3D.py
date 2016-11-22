#####################################################################################################

# Python script which computes a real Fourier transform 
# The idea was to find a way to compute the "roughness" of the white matter
# which we visually thought to increase with the age, but it hasn't been conclusive.

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
T = 278
''' #for testing
X = 50
Y = 50
Z = 50
T = 3
'''
train = [None]*T
data = [None]*T

labels = np.genfromtxt('../data/targets.csv', delimiter="\n")

status = [None]*T
STATUS = ["diseased","healthy"]
for i in range(T):
    status[i] = STATUS[int(labels[i])]

colormap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['black','white'],256)

# MASK TO TAKE AWAY 0 FREQUENCIES
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


# INITIALIZE WRITING
featuresFile = open('../results/fourierTrain.csv','w')

# LOOP OVER ALL PICTURES

for i in xrange(T):
    print("Computing fourier of train"+str(i+1)+"...")

    train[i] = nib.load("../data/set_train/train_"+str(i+1)+".nii")

    data[i] = train[i].get_data()

    data_xy = np.squeeze(np.mean(data[i],axis=2),axis=2)
    data_xz = np.squeeze(np.mean(data[i],axis=1),axis=2)
    data_yz = np.squeeze(np.mean(data[i],axis=0),axis=2)

    data_xy = np.fft.fft2(data_xy)
    data_xy = np.log(np.abs(np.fft.fftshift(data_xy))**2)
    data_xy = np.multiply(data_xy,frequencies_mask_xy)
    row_totals = []
    row_totals = [ sum(x) for x in data_xy ]
    plt.hist(row_totals,bins=50)
    plt.savefig("../plots/fourier1D_xy_"+status[i]+"_train"+str(i)+".png")
    # del row_totals
    # val=0
    # plt.plot(row_totals, np.zeros_like(row_totals) + val, 'x')
    # fouriers = plt.figure(1)
    # plt.imshow(data_xy,cmap=colormap,
    #                 origin='lower')
    # plt.title("Fourier in XY ("+status[i]+")")
    # plt.savefig("../plots/fourier1D_xy_"+status[i]+"_train"+str(i)+".png")

    data_xz = np.fft.fft2(data_xz)
    '''
    fouriers = plt.figure(1)
    data_xz = np.log(np.abs(np.fft.fftshift(data_xz))**2)
    data_xz = np.multiply(data_xz,frequencies_mask_xz)
    plt.imshow(data_xz)
    plt.title("Fourier in XZ (age "+str(int(ages[i]))+")")
    fouriers.savefig("../plots/fourier_xz_age"+str(int(ages[i]))+"_train"+str(i)+".png")
    '''
    
    data_yz = np.fft.fft2(data_yz)
    '''
    fouriers = plt.figure(2)
    data_yz = np.log(np.abs(np.fft.fftshift(data_yz))**2)
    data_yz = np.multiply(data_yz,frequencies_mask_yz)
    plt.imshow(data_yz)
    plt.title("Fourier in YZ (age "+str(int(ages[i]))+")")
    fouriers.savefig("../plots/fourier_yz_age"+str(int(ages[i]))+"_train"+str(i)+".png")
    '''
    
    #print(data_xy[1])
    #data_final = np.concatenate((data_xy,data_xz,data_yz))
    #np.savetext('../results/fourierTrain.csv',data_final,delimiter=',')
    
    '''
    featuresFile.write(data_xy)
    featuresFile.write(data_xz)
    featuresFile.write(data_yz)
    featuresFile.write('\n')
    '''
    
featuresFile.close()
    
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

