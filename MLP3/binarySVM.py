########################################################################
#               Ridge + binary SVM classification                      #
########################################################################

TEST=138
TRAIN=278

CLISTS = [[0.0035,0.0014,0.0002]]
kernel = 'linear'

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing

def svmClassification(features, targets, C=1, kernel='rbf', degree=3, gamma='auto', decision_function_shape=None, prediction=False, toPredict=np.empty(1, dtype=int)):
    modelSVM = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, decision_function_shape=decision_function_shape, probability=True)
    modelSVM.fit(features, targets)
    scoreFinal = modelSVM.score(features, targets)
    #prediction
    if prediction==True:
        predictionOutput = modelSVM.predict(toPredict)
        prob = modelSVM.predict_proba(toPredict)
        return {'SV': modelSVM.support_vectors_, 'SV indices': modelSVM.support_, 'SV repartition': modelSVM.n_support_, 'Coefficients': modelSVM.coef_, 'Intercept': modelSVM.intercept_, 'Score': scoreFinal, 'Predicted': predictionOutput, 'Probabilities': prob}
    else:
        return {'SV': modelSVM.support_vectors_, 'SV indices': modelSVM.support_, 'SV repartition': modelSVM.n_support_, 'Coefficients': modelSVM.coef_, 'Intercept': modelSVM.intercept_, 'Score': scoreFinal}

### Features for the reduction
sectionFeatures = np.genfromtxt('features/train_section_features.csv', delimiter=",")
segmentFeatures = np.genfromtxt('features/train_segment_features.csv', delimiter=",")
features = np.concatenate([sectionFeatures, segmentFeatures],1)
featuresScaled = preprocessing.scale(features)

### Features for the prediction
sectionFeatures = np.genfromtxt('features/test_section_features.csv', delimiter=",")
segmentFeatures = np.genfromtxt('features/test_segment_features.csv', delimiter=",")
toPredictFeatures = np.concatenate([sectionFeatures,segmentFeatures],1)
toPredictFeaturesScaled = preprocessing.scale(toPredictFeatures)

### Targets
targets = np.genfromtxt('data/targets.csv', delimiter=",")

for clist in CLISTS:
    ### Binary SVM
    SVMpredictions = [[],[],[]]
    SVMprobab = [[],[],[]]
    for charac in [0,1,2]:
        print("Start SVM classification of characteristic "+str(charac)+" with c = "+str(clist[charac]))
        SVMResults = svmClassification(featuresScaled, targets=targets[:,charac], C=clist[charac], kernel=kernel, \
        degree=2, gamma='auto', decision_function_shape=None, prediction=True, toPredict=toPredictFeaturesScaled)
        SVMpredictions[charac] = SVMResults['Predicted']
        '''SVMprobab[charac] = SVMResults['Probabilities']'''

    ### Write in a csv file
    result = open('results/binarySVM_'+str(clist)+'.csv','w')
    result.write("ID,Sample,Label,Predicted"+"\n")
    characNames = [',gender,',',age,',',health,']
    for id in range(TEST*3):
        result.write(str(id)+','+str(id/3)+characNames[id%3]+str(bool(SVMpredictions[id%3][id/3])).upper()+"\n")
    result.close()
