########################################################################
#               Ridge + binary SVM classification                      #
########################################################################

TEST=138
TRAIN=278

ALPHA = 0.4 #for Ridge

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

### Features for the prediction
sectionFeatures = np.genfromtxt('features/test_section_features.csv', delimiter=",")
segmentFeatures = np.genfromtxt('features/test_segment_features.csv', delimiter=",")
toPredictFeatures = np.concatenate([sectionFeatures,segmentFeatures],1)
toPredictFeaturesScaled = preprocessing.scale(toPredictFeatures)

### Targets for Ridge and SVM
targets = np.genfromtxt('data/targets.csv', delimiter=",")

### Ridge regression
print("Start Ridge regression with alpha = "+str(ALPHA))
ridgeResults = ridgeRegression([ALPHA], features, targets, True, toPredict=toPredictFeatures)
ridgePredictions = ridgeResults['Predicted']

### Binary SVM
SVMpredictions = [[],[],[]]
SVMprobab = [[],[],[]]
for charac in [0,1,2]:
    print("Start SVM classification of characteristic "+str(charac)+" with c = "+str(CLIST[charac]))
    SVMResults = svmClassification(featuresScaled, targets=targets[:,charac], C=CLIST[charac], kernel=kernel, \
    degree=2, gamma='auto', decision_function_shape=None, prediction=True, toPredict=toPredictFeaturesScaled)
    SVMpredictions[charac] = SVMResults['Predicted']
    SVMprobab[charac] = SVMResults['Probabilities']

### Average
average = []
for id in range(3*TEST):
    if round(SVMprobab[id%3][id/3][1])!=SVMpredictions[id%3][id/3]: # the case when the SVM probability and prediction disagree
        SVMprobab[id%3][id/3][1] = 0.5
    average.append(WEIGHT_SVM * SVMprobab[id%3][id/3][1] + WEIGHT_RIDGE * ridgePredictions[id/3][id%3])
    
    '''print(str(bool(SVMpredictions[id%3][id/3])==(ridgePredictions[id/3][id%3]>=0.5)).upper()+'   '+str(id%3)+' '+str(id/3)+' SVM: '+str(bool(SVMpredictions[id%3][id/3])).upper()+' proba: '+str(SVMprobab[id%3][id/3][1])+' ('+str(round(SVMprobab[id%3][id/3][1])==SVMpredictions[id%3][id/3])+') Ridge: '+str(ridgePredictions[id/3][id%3]>=0.5).upper()+' '+str(ridgePredictions[id/3][id%3]))
    print("   ->"+str(average[id]>=0.5).upper()+' '+str(average[id]))'''

### Write in a csv file
result = open('results/average_SVM_'+str(WEIGHT_SVM)+'_RIDGE_'+str(WEIGHT_RIDGE)+'.csv','w')
result.write("ID,Sample,Label,Predicted"+"\n")
characNames = [',gender,',',age,',',health,']
for id in range(TEST*3):
    result.write(str(id)+','+str(id/3)+characNames[id%3]+str(average[id]>=0.5).upper()+"\n")
result.close()
