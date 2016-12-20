########################################################################

# Python script to perform segmentation

########################################################################

#''' (remove/add # to switch)
TRAIN=269
X = 176
Y = 208
Z = 176
''' #for training with smaller values
TRAIN=10
X = 176
Y = 208
Z = 176
#'''

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import svm
from scipy import ndimage
from detect_peaks import detect_peaks

images = [None]*TRAIN
data = [None]*TRAIN
genders = np.genfromtxt('segment/correct_genders.csv', delimiter="\n")
colors = ['blue' if g==0 else 'red' for g in genders[0:TRAIN]]

volumes = np.genfromtxt('segment/correct_volumes.csv', delimiter="\n")
frontiers = np.genfromtxt('segment/correct_frontiers.csv', delimiter="\n")
features = np.concatenate([np.reshape(volumes,[-1,1]),np.reshape(frontiers,[-1,1])],1)

def svmclassification(features, targets, C=1, kernel='rbf', degree=3, gamma='auto', decision_function_shape=None, prediction=False, toPredict=np.empty(1, dtype=int)):

    modelSVM = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, decision_function_shape=decision_function_shape)

    modelSVM.fit(features, targets)

    dist = modelSVM.decision_function(features)

    scoreFinal = modelSVM.score(features, targets)

    print("SV repartition: {0} Score: {1}".format(modelSVM.n_support_, scoreFinal))

    # Prediction
    if prediction==True:
        predictionOutput = (modelSVM.predict(toPredict))
        return {'SV': modelSVM.support_vectors_, 'SV indices': modelSVM.support_, 'SV repartition': modelSVM.n_support_, 'SV coefficients': modelSVM.dual_coef_, 'Coefficients': modelSVM.coef_, 'Intercept': modelSVM.intercept_, 'Score': scoreFinal, 'Predicted': predictionOutput}
    else:
        return {'SV': modelSVM.support_vectors_, 'SV indices': modelSVM.support_, 'SV repartition': modelSVM.n_support_, 'SV coefficients': modelSVM.dual_coef_, 'Coefficients': modelSVM.coef_, 'Intercept': modelSVM.intercept_, 'Score': scoreFinal}
plt.figure(1)
plt.scatter(volumes, frontiers, c=colors, alpha=0.5)

kernel='linear'
clist = [0.000001,100]
for c in clist:
    results = svmclassification(features, targets=genders, C=c, kernel=kernel, degree=3, gamma='auto', decision_function_shape=None, prediction=False, toPredict=None)
    coef0 = results['Coefficients'][0,0]
    coef1 = results['Coefficients'][0,1]
    intercept = results['Intercept'][0]
    print(coef0)
    print(coef1)
    print(intercept)

    plt.plot(np.linspace(min(volumes),max(volumes)), [-coef0/coef1*x - intercept/coef1 for x in np.linspace(min(volumes),max(volumes))])
    plt.plot(np.linspace(min(volumes),max(volumes)), [-coef0/coef1*x - intercept/coef1 +1/coef1 for x in np.linspace(min(volumes),max(volumes))])
    plt.plot(np.linspace(min(volumes),max(volumes)), [-coef0/coef1*x - intercept/coef1 -1/coef1 for x in np.linspace(min(volumes),max(volumes))])

plt.show()
