########################################################################

# Python script which plots an histogram for each test brain

########################################################################

import numpy as np
import nibabel as nib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

#import sklearn as sk
#from sklearn.linear_model import Lasso

import os, sys

#'''
X = 176
Y = 208
Z = 176
T = 138
''' #for testing
X = 10
Y = 10
Z = 10
T = 2
#'''
test = [None]*T
data = [None]*T

for i in range(int(sys.argv[1]),int(sys.argv[2])):
    print("plotting histogram of test"+str(i+1)+"...")
    #relative path
    test[i] = nib.load("../data/set_test/test_"+str(i+1)+".nii")

    data[i] = test[i].get_data()

    # store all the non zero values in a 1D list
    intList=[2500]
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if data[i][x,y,z]!=0:
                    intList.append(int(data[i][x,y,z]))

    # plot and save figure
    histo = plt.figure(0)
    binwidth = 10
    bins = range(0, 2000 + binwidth, binwidth)
    plt.hist(intList, bins=bins)
    plt.axis([0, 2000, 0, 40000])
    plt.title("Histogram of test"+str(i+1))
    plt.xlabel("Values of the 3D brain")
    plt.ylabel("Frequencies")
    histo.savefig("../plots/plot_histo_test"+str(i+1)+".png")
    plt.clf() #close figure
