########################################################################
#               Ridge regression for different Alphas                  #
########################################################################

TEST=138
TRAIN=278

ALPHAS = [[0.4, 0.4, 0.4]]
if (ALPHAS[0][1]==ALPHAS[0][2]):
    key = ALPHAS[0][0]
    keytxt = "gender"
elif (ALPHAS[0][0]==ALPHAS[0][2]):
    key = ALPHAS[0][1]
    keytxt = "age"
else:
    key = ALPHAS[0][2]
    keytxt = "health"

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

### Features for the prediction
sectionFeatures = np.genfromtxt('../features/test_section_features.csv', delimiter=",")
segmentFeatures = np.genfromtxt('../features/test_segment_features.csv', delimiter=",")
toPredictFeatures = np.concatenate([sectionFeatures,segmentFeatures],1)

### Targets
targets_gender = np.genfromtxt('../data/targets_gender.csv', delimiter=",")
targets_age = np.genfromtxt('../data/targets_age.csv', delimiter=",")
targets_health = np.genfromtxt('../data/targets_health.csv', delimiter=",")

### Cross Validation ###
mask = np.ones((TRAIN,1),dtype=bool)
error = []
perfArray = np.empty((TRAIN,3))

for alpha in ALPHAS:
    for idx in range(TRAIN):

        mask[idx] = 0
        print("Start Ridge regression for " + keytxt + " CV with alpha = "+str(key))

        ### Ridge regression for gender
        ridgeResults = ridgeRegression(alphas=[alpha[0]], features=features[mask], target=targets_gender[mask], prediction=True, toPredict=toPredictFeatures[not mask])
        ridgePredictionsGender = ridgeResults['Predicted']
        ridgePredictionsGender[[i for i in probaRidge if i>=0.5]] = 1 
        ridgePredictionsGender[[i for i in probaRidge if i<0.5]] = 0

        ### Ridge regression for age
        ridgeResults = ridgeRegression(alphas=[alpha[1]], features=features[mask], target=targets_age[mask], prediction=True, toPredict=toPredictFeatures[not mask])
        ridgePredictionsAge = ridgeResults['Predicted']
        ridgePredictionsAge[[i for i in probaRidge if i>=0.5]] = 1 
        ridgePredictionsAge[[i for i in probaRidge if i<0.5]] = 0

        ### Ridge regression for health
        ridgeResults = ridgeRegression(alphas=[alpha[2]], features=features[mask], target=targets_health[mask], prediction=True, toPredict=toPredictFeatures[not mask])
        ridgePredictionsHealth = ridgeResults['Predicted']   
        ridgePredictionsHealth[[i for i in probaRidge if i>=0.5]] = 1 
        ridgePredictionsHealth[[i for i in probaRidge if i<0.5]] = 0

        perfArray[idx,0] = int((ridgePredictionsGender==targets_gender[not mask]))
        perfArray[idx,1] = int((ridgePredictionsAge==targets_age[not mask]))
        perfArray[idx,2] = int((ridgePredictionsHealth==targets_health[not mask]))

        mask[idx] = 1 # reset

    error.append(TRAIN-np.sum(perfArray))

print(keytxt)
print(ALPHAS)
print(error)

### TODO: compare the perfs in a meaningful way. Right now they're just sums of booleans.
