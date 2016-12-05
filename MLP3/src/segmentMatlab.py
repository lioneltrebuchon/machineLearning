########################################################################

# Python script to output result from SVM classification

########################################################################

TEST=138
TRAIN=278

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import svm

import nipype.interfaces.spm as spm
import nipype.interfaces.matlab as matlab

print("lol1")
seg = spm.Segment(matlab_cmd = matlab.MatlabCommand(mfile=False))
print("lol2")
seg.inputs.data = '/local/set_train/train_1.nii'
#seg.inputs.data = nib.load("../data/set_"+imagesSet+"/"+imagesSet+"_"+str(i+1)+".nii")
print("lol3")
seg.run()
print("lol4")


#images[i] = nib.load("../data/set_"+imagesSet+"/"+imagesSet+"_"+str(i+1)+".nii")
#data[i] = images[i].get_data()

