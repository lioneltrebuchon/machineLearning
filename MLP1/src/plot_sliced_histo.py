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
sizeCutX = int(round(X/nCuts))
sizeCutY = int(round(Y/nCuts))
sizeCutZ = int(round(Z/nCuts))
for i in xrange(T):
    print("plotting histogram of train"+str(i+1)+"...")
    #relative path
    train[i] = nib.load("../data/set_train/train_"+str(i+1)+".nii")

    data[i] = train[i].get_data()
    
    for cut in xrange(nCuts):
        # store all the non zero values in a 1D list
        intList=[]
        print("plotting x-histogram of slice"+str(cut)+"...") 
        for x in range((cut-1)*sizeCutX,cut*sizeCutX):
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
        histo.savefig("../plots/histoXcut_age"+str(int(ages[i]))+"_train"+str(i+1)+"_cut"+str(cut)+".png")
        plt.clf() #close figure
        del intList[:]

    
    for cut in xrange(nCuts):
        # store all the non zero values in a 1D list
        intList=[]
        print("plotting y-histogram of slice"+str(cut)+"...") 
        for y in range((cut-1)*sizeCutY,cut*sizeCutY):
            for x in range(X):
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
        histo.savefig("../plots/histoYcut_age"+str(int(ages[i]))+"_train"+str(i+1)+"_cut"+str(cut)+".png")
        plt.clf() #close figure
        del intList[:]

    for cut in xrange(nCuts):
        # store all the non zero values in a 1D list
        intList=[]
        print("plotting y-histogram of slice"+str(cut)+"...") 
        for z in range((cut-1)*sizeCutZ,cut*sizeCutZ):
            for x in range(X):
                for y in range(y):
                    if data[i][x,y,z]!=0:
                        intList.append(int(data[i][x,y,z]))
        # plot and save figure
        histo = plt.figure(cut)
        plt.hist(intList, 200)
        plt.axis([0, 2800, 0, 6000])
        plt.title("Histogram of Train "+str(i+1)+"(age "+str(int(ages[i]))+")."+"Cut nr."+str(cut))
        plt.xlabel("Values of the 3D brain")
        plt.ylabel("Frequencies")
        histo.savefig("../plots/histoZcut_age"+str(int(ages[i]))+"_train"+str(i+1)+"_cut"+str(cut)+".png")
        plt.clf() #close figure
        del intList[:]

for i in range(len(peakIndexes)):
        peaks.append([values[1][peakIndexes[i]],values[0][peakIndexes[i]]])
        print("Peak "+str(i)+": ")
        print(peaks[i])
        peaksFile.write(", "+str(peaks[i]))
