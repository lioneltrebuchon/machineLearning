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

for i in range(T):
	print("plotting histogram of train"+str(i+1)+"...")
	#relative path
	train[i] = nib.load("../data/set_train/train_"+str(i+1)+".nii")

	data[i] = train[i].get_data()

	# store all the non zero values in a 1D list
	intList=[]
	for x in range(X):
		for y in range(Y):
		    for z in range(Z):
				if data[i][x,y,z]!=0:
					intList.append(int(data[i][x,y,z]))

	# plot and save figure
	histo = plt.figure(0)
	plt.hist(intList, 200)
	plt.axis([0, 3000, 0, 50000])
	plt.title("Histogram of Train"+str(i+1)+" (age "+str(int(ages[i]))+")")
	plt.xlabel("Values of the 3D brain")
	plt.ylabel("Frequencies")
	histo.savefig("../plots/plot_histo_age"+str(int(ages[i]))+"_train"+str(i+1)+".png")
	plt.clf() #close figure
