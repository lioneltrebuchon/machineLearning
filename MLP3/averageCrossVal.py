########################################################################
#               Ridge + binary SVM classification                      #
########################################################################

TEST=138
TRAIN=278

ALPHALIST = [0.75,0.75,0.75] #for Ridge

CLIST = [0.0035,0.0014,0.0002] #for SVM
kernel = 'linear'

WEIGHT_RIDGE = 0.5
WEIGHT_SVM = 1 - WEIGHT_RIDGE

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import linear_model
from sklearn import preprocessing

def ridgeRegression(alphas, features, target, prediction=False, toPredict=np.empty(1, dtype=int)):
    modelRidge = linear_model.RidgeCV(alphas, normalize=True, cv=None)
    result = modelRidge.fit(features, target)
    scoreFinal = modelRidge.score(features, target)
    if prediction==True:
        predictionOutput = modelRidge.predict(toPredict)
        return {'Coefficient': modelRidge.coef_, 'Alpha': modelRidge.alpha_, 'Score': scoreFinal, 'Intercept': modelRidge.intercept_, 'Predicted': predictionOutput}
    else:
        return {'Coefficient': modelRidge.coef_, 'Alpha': modelRidge.alpha_, 'Score': scoreFinal, 'Intercept': modelRidge.intercept_}

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

'''
### Features for the prediction
sectionFeatures = np.genfromtxt('features/test_section_features.csv', delimiter=",")
segmentFeatures = np.genfromtxt('features/test_segment_features.csv', delimiter=",")
toPredictFeatures = np.concatenate([sectionFeatures,segmentFeatures],1)
toPredictFeaturesScaled = preprocessing.scale(toPredictFeatures)
'''

### Targets for Ridge and SVM
targets = np.genfromtxt('data/targets.csv', delimiter=",")

mask = np.ones((TRAIN),dtype=bool)
error = 0

### Cross validation
for idx in range(TRAIN):
    mask[idx] = False

    print("id "+str(idx))
    for charac in [0,1,2]:

        ### Ridge regression
        print("Start Ridge regression of characteristic "+str(charac)+" with alpha = "+str(ALPHALIST[charac]))
        ridgeResults = ridgeRegression([ALPHALIST[charac]], features[mask], targets[mask,charac], True, toPredict=features[idx].reshape(1,-1))
        ridgePrediction = ridgeResults['Predicted'][0]

        ### Binary SVM
        print("Start SVM classification of characteristic "+str(charac)+" with c = "+str(CLIST[charac]))
        SVMResults = svmClassification(featuresScaled[mask], targets=targets[mask,charac], C=CLIST[charac], kernel=kernel, \
        degree=2, gamma='auto', decision_function_shape=None, prediction=True, toPredict=featuresScaled[idx].reshape(1,-1))
        SVMPrediction = SVMResults['Predicted'][0]
        SVMProbab = SVMResults['Probabilities'][0]
        if round(SVMProbab[1])!=SVMPrediction: # the case when the SVM probability and prediction disagree
            SVMProbab[1] = 0.5
            print('fail')

        ### Average
        average = WEIGHT_SVM * SVMProbab[1] + WEIGHT_RIDGE * ridgePrediction
        error += (targets[idx,charac]-int(average >= 0.5))**2

    mask[idx] = True # reset

error /= (3*TRAIN)
print("error = "+str(error))
