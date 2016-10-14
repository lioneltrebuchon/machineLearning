########################################################################

# Python script which plots an histogram for each training brain

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
xyz = [X,Y,Z]
T = 278
''' #for testing
X = 10
Y = 10
Z = 10
T = 2
'''
T = 2
nCuts = 10

train = [None]*T
data = [None]*T

ages = np.genfromtxt('../data/targets.csv', delimiter="\n")

#for axis in [1,2,3]
sizeCut = int(round(X/nCuts))
for i in xrange(T):
    print("plotting histogram of train"+str(i+1)+"...")
    #relative path
    train[i] = nib.load("../data/set_train/train_"+str(i+1)+".nii")

    data[i] = train[i].get_data()
    
    for cut in xrange(nCuts):
        # store all the non zero values in a 1D list
        intList=[]
        print("plotting histogram of slice"+str(cut)+"...") 
        for x in range((cut-1)*sizeCut,cut*sizeCut):
            for y in range(Y):
                for z in range(Z):
                    if data[i][x,y,z]!=0:
                        intList.append(int(data[i][x,y,z]))
        # plot and save figure
        histo = plt.figure(cut)
        plt.hist(intList, 200)
        plt.axis([0, 2800, 0, 6000])
        plt.title("Histogram of Train "+str(i+1)+"(age "+str(int(ages[i]))+")."+"Cut nr."+str(cut))
        plt.xlabel("Values of the 3D brain")
        plt.ylabel("Frequencies")
        histo.savefig("../plots/plot_histo_age"+str(int(ages[i]))+"_train"+str(i+1)+"_cut"+str(cut)+".png")
        plt.clf() #close figure
        del intList[:]
