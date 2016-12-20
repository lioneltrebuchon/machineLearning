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
sectionFeatures = np.genfromtxt('features/train_section_features.csv', delimiter=",")
segmentFeatures = np.genfromtxt('features/train_segment_features.csv', delimiter=",")
features = np.concatenate([sectionFeatures, segmentFeatures],1)

### Features for the prediction
sectionFeatures = np.genfromtxt('features/test_section_features.csv', delimiter=",")
segmentFeatures = np.genfromtxt('features/test_segment_features.csv', delimiter=",")
toPredictFeatures = np.concatenate([sectionFeatures,segmentFeatures],1)

### Targets
targets_gender = np.genfromtxt('data/targets_gender.csv', delimiter=",")
targets_age = np.genfromtxt('data/targets_age.csv', delimiter=",")
targets_health = np.genfromtxt('data/targets_health.csv', delimiter=",")

for alpha in ALPHAS:
    
    print("Start Ridge regression for " + keytxt + " CV with alpha = "+str(key))

    ### Ridge regression for gender
    ridgeResults = ridgeRegression([alpha[0]], features, targets_gender, True, toPredict=toPredictFeatures)
    ridgePredictionsGender = ridgeResults['Predicted']

    ### Ridge regression for age
    ridgeResults = ridgeRegression([alpha[1]], features, targets_age, True, toPredict=toPredictFeatures)
    ridgePredictionsAge = ridgeResults['Predicted']

    ### Ridge regression for health
    ridgeResults = ridgeRegression([alpha[2]], features, targets_health, True, toPredict=toPredictFeatures)
    ridgePredictionsHealth = ridgeResults['Predicted']

    ### Write in a csv file
    result = open('results/ridgeBinary_'+keytxt+str(key)+'.csv','w')
    result.write("ID,Sample,Label,Predicted"+"\n")
    characNames = [',gender,',',age,',',health,']
    for id in range(TEST*3):
        result.write(str(id)+','+str(id/3)+characNames[id%3]+str(ridgePredictionsGender[id/3]>=0.5).upper()+"\n")
        result.write(str(id)+','+str(id/3)+characNames[id%3]+str(ridgePredictionsAge[id/3]>=0.5).upper()+"\n")
        result.write(str(id)+','+str(id/3)+characNames[id%3]+str(ridgePredictionsHealth[id/3]>=0.5).upper()+"\n")
    result.close()
