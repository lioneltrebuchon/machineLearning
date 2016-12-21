########################################################################
#               Ridge regression cross val for different Alphas        #
########################################################################

TEST=138
TRAIN=278

ALPHAS = [0.65,0.7,0.75,0.8,0.85]

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
sectionFeatures = np.genfromtxt('../features/train_section_features.csv', delimiter=",")
segmentFeatures = np.genfromtxt('../features/train_segment_features.csv', delimiter=",")
features = np.concatenate([sectionFeatures, segmentFeatures],1)

'''
### Features for the prediction
sectionFeatures = np.genfromtxt('../features/test_section_features.csv', delimiter=",")
segmentFeatures = np.genfromtxt('../features/test_segment_features.csv', delimiter=",")
toPredictFeatures = np.concatenate([sectionFeatures,segmentFeatures],1)
'''

### Targets
targets = np.genfromtxt('../data/targets.csv', delimiter=",")

### Cross Validation
mask = np.ones((TRAIN),dtype=bool)

for alpha in ALPHAS:
    print("Start cross validation with alpha = "+str(alpha))
    error = 0
    for idx in range(TRAIN):
        mask[idx] = False

        #print("id "+str(idx))
        ridgeResults = ridgeRegression([alpha], features[mask], targets[mask], True, toPredict=features[idx].reshape(1, -1))
        ridgePredictions = ridgeResults['Predicted']
        roundedPredictions = np.array(ridgePredictions > 0.5,dtype=int)
        error += np.sum((roundedPredictions-targets[idx])**2)

        mask[idx] = True # reset
    error /= (3*TRAIN)
    print("error = "+str(error))
