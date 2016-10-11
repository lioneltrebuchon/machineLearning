import numpy as np
import nibabel as nib
import matplotlib as plt
#import sklearn as sk   #Narrive pas a le DL car je nai pas reussi a installer le package scipy
#from sklearn.linear_model import Lasso
import os, sys

X = 175
Y = 207
Z = 175

train1 = nib.load("../set_train/train_1.nii")
data = train1.get_data()

#compute the max intensity
'''
biggest = 0
for x in range(X):
    for y in range(Y):
        for z in range(Z):
            if data[x, y, z] > biggest:
                biggest = data[x, y, z]
print("Biggest intensity = " + str(biggest))
'''

#plot the brain projected on x and y axis

for x in range(X):
    for y in range(Y):
	max_z[x,y] = 0
        for z in range(Z):
            if data[x, y, z] > max_z[x,y]:
		max_z[x,y] = data[x, y, z]
plt.plot(max_z)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#plot the brain projected on x and y axis

for x in range(X):
    for z in range(Z):
	max_y[x,z] = 0
        for y in range(Y):
            if data[x, y, z] > max_y[x,z]:
		max_y[x,z] = data[x, y, z]
plt.plot(max_y)
plt.xlabel('x')
plt.ylabel('z')
plt.show()

#plot the brain projected on x and y axis

for z in range(Z):
    for y in range(Y):
	max_x[z,y] = 0
        for x in range(X):
            if data[x, y, z] > max_x[z,y]:
		max_x[z,y] = data[x, y, z]
plt.plot(max_x)
plt.xlabel('z')
plt.ylabel('y')
plt.show()
