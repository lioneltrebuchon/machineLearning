########################################################################
#               SVM classification with cross validation               #
########################################################################

TEST=138
TRAIN=278

C = [0.0005, 0.001, 0.0015]
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

'''
### Features for the prediction
sectionFeatures = np.genfromtxt('features/test_section_features.csv', delimiter=",")
segmentFeatures = np.genfromtxt('features/test_segment_features.csv', delimiter=",")
toPredictFeatures = np.concatenate([sectionFeatures,segmentFeatures],1)
toPredictFeaturesScaled = preprocessing.scale(toPredictFeatures)
'''

### Targets
targets = np.genfromtxt('data/targets.csv', delimiter=",")
multiTargets = targets[:,0]+2*targets[:,1]+4*targets[:,2]


for c in C:

    ### Cross Validation
    print("Start cross validation with c = "+str(c))

    mask = np.ones((TRAIN),dtype=bool)
    error = 0

    for idx in range(TRAIN):
        mask[idx] = False

        print("id "+str(idx))
        SVMResults = svmClassification(featuresScaled[mask], targets=multiTargets[mask], C=c, kernel=kernel, \
        degree=2, gamma='auto', decision_function_shape=None, prediction=True, toPredict=featuresScaled[idx].reshape(1, -1))
        SVMpredictions = SVMResults['Predicted']
        '''SVMprobab = SVMResults['Probabilities']'''

        prediction = SVMpredictions[0]
        for charac in [0,1,2]:
            error += (targets[idx,charac]-prediction%2)**2
            prediction -= prediction%2
            prediction /= 2

        mask[idx] = True # reset
    error /= (3*TRAIN)
    print("error = "+str(error))
