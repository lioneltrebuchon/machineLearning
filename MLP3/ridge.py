########################################################################
#               Ridge regression for different Alphas                  #
########################################################################

TEST=138
TRAIN=278

ALPHALISTS = [[0.4,0.4,0.4]]

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

### Features for the prediction
sectionFeatures = np.genfromtxt('features/test_section_features.csv', delimiter=",")
segmentFeatures = np.genfromtxt('features/test_segment_features.csv', delimiter=",")
toPredictFeatures = np.concatenate([sectionFeatures,segmentFeatures],1)

### Targets
targets = np.genfromtxt('data/targets.csv', delimiter=",")

for alphaList in ALPHALISTS:
    ### Ridge regressions
    ridgePredictions = [[],[],[]]
    for charac in [0,1,2]:
        print("Start Ridge regression of characteristic "+str(charac)+" with alpha = "+str(alphaList[charac]))
        ridgeResults = ridgeRegression([alphaList[charac]], features, targets[:,charac], True, toPredict=toPredictFeatures)
        ridgePredictions[charac] = ridgeResults['Predicted']

    ### Write in a csv file
    result = open('results/ridge_'+str(alphaList)+'.csv','w')
    result.write("ID,Sample,Label,Predicted"+"\n")
    characNames = [',gender,',',age,',',health,']
    for id in range(TEST*3):
        result.write(str(id)+','+str(id/3)+characNames[id%3]+str(ridgePredictions[id%3][id/3]>=0.5).upper()+"\n")
    result.close()
