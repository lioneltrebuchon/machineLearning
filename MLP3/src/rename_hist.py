import numpy as np
import nibabel as nib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

#import sklearn as sk
#from sklearn.linear_model import Lasso

import os, sys

targetsAge = np.genfromtxt('../data/targets_age.csv', delimiter="\n")
targetsGender = np.genfromtxt('../data/targets_gender.csv', delimiter="\n")
targetsHealth = np.genfromtxt('../data/targets_health.csv', delimiter="\n")
path = '/nas/bbidot/hist_test/'

for i in range(278):
    if targetsHealth[i]==0:
        filename = path+'plot_histo_'
        if targetsAge[i]==0:
            age = 'old'
        else:
            age = 'young'
        if targetsGender[i]==0:
            gender = 'male'
        else:
            gender = 'fem'
        filename1 = filename+'diseased_train'+str(i+1)+'.png'
        print(filename1)
        filename2 = filename+'sick_'+age+'_'+gender+'_train'+str(i+1)+'.png'
        os.rename(filename1 , filename2)
    else:
        filename = path+'plot_histo_healthy'
        if targetsAge[i]==0:
            age = 'old'
        else:
            age = 'young'
        if targetsGender[i]==0:
            gender = 'male'
        else:
            gender = 'fem'
        filename1 = filename+'_train'+str(i+1)+'.png' 
        filename2 = filename+'_'+age+'_'+gender+'_train'+str(i+1)+'.png'
        os.rename(filename1 , filename2)
    
