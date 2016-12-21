########################################################################
#               Ridge regression with cross validation                 #
########################################################################

TEST=138
TRAIN=278

ALPHALISTS = [[0.75,0.75,0.75]]

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import linear_model

def ridgeRegression(alphas, features, target, prediction=False, toPredict=np.empty(1, dtype=int)):
    modelRidge = linear_model.RidgeCV(alphas, normalize=True, cv=None)
    result = modelRidge.fit(features, target)
    scoreFinal = modelRidge.score(features, target)
    if prediction==True:
        predictionOutput = modelRidge.predict(toPredict)
        return {'Coefficient': modelRidge.coef_, 'Alpha': modelRidge.alpha_, 'Score': scoreFinal, 'Intercept': modelRidge.intercept_, 'Predicted': predictionOutput}
    else:
        return {'Coefficient': modelRidge.coef_, 'Alpha': modelRidge.alpha_, 'Score': scoreFinal, 'Intercept': modelRidge.intercept_}

### Features for the reduction
sectionFeatures = np.genfromtxt('features/train_section_features.csv', delimiter=",")
segmentFeatures = np.genfromtxt('features/train_segment_features.csv', delimiter=",")
features = np.concatenate([sectionFeatures, segmentFeatures],1)

'''
### Features for the prediction
sectionFeatures = np.genfromtxt('features/test_section_features.csv', delimiter=",")
segmentFeatures = np.genfromtxt('features/test_segment_features.csv', delimiter=",")
toPredictFeatures = np.concatenate([sectionFeatures,segmentFeatures],1)
'''

### Targets
targets = np.genfromtxt('data/targets.csv', delimiter=",")

for alphaList in ALPHALISTS:

    ### Cross Validation
    print("Start cross validation with alphaList = "+str(alphaList))

    mask = np.ones((TRAIN),dtype=bool)
    error = 0

    for idx in range(TRAIN):
        mask[idx] = False

        print("id "+str(idx))
        for charac in [0,1,2]:
            ridgeResults = ridgeRegression([alphaList[charac]], features[mask], targets[mask,charac], True, toPredict=features[idx].reshape(1,-1))
            prediction = ridgeResults['Predicted'][0]
            error += (targets[idx,charac]-int(prediction >= 0.5))**2

        mask[idx] = True # reset
    error /= (3*TRAIN)
    print("error = "+str(error))
