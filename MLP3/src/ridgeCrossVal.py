########################################################################

# Python script to output result from Ridge regression

########################################################################

TEST=138
TRAIN=278

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.cross_validation import cross_val_score


### INPUTS: FEATURES AND TARGETS ### 
features = np.genfromtxt('../results/sliceTrainFeatures32.csv', delimiter=",")
target = np.genfromtxt('../data/targets.csv', delimiter="\n")
toPredictFeatures = np.genfromtxt('../results/train_section_features.csv', delimiter=",")


### RIDGE REGRESSION ###
def ridgeRegression(alphas, features, target, prediction=False, toPredict=np.empty(1, dtype=int)):
    modelRidge = linear_model.RidgeCV(alphas, normalize=True, cv=10)
    result = modelRidge.fit(features, target)
    scoreFinal = modelRidge.score(features,target)
    # Prediction
    if prediction==True:
        predictionOutput = modelRidge.predict(toPredict)
        return {'Coefficient': modelRidge.coef_, 'Alpha': modelRidge.alpha_, 'Score': scoreFinal, 'Intercept': modelRidge.intercept_, 'Predicted': predictionOutput}
    else:
        return {'Coefficient': modelRidge.coef_, 'Alpha': modelRidge.alpha_, 'Score': scoreFinal, 'Intercept': modelRidge.intercept_}

### SET PARAMETERS FOR RIDGE ###
alphas = np.linspace(1.8*10**8, 2.2*10**8, 20)

### RUN ###
mask = np.ones((TRAIN,1),dtype=bool)
#probaRidge = np.zeros((TRAIN,3))
perf = np.zeroes((TRAIN,1))
print("Start SVM (C="+str(c)+", kernel:"+kernel+") and Ridge regression (alpha="+str(alpha)+")")
for alpha in alphas
    for idx in range(TRAIN):
        mask[idx] = 0
        trainedRidge = ridgeRegression(alphas=[alpha], features=features[mask], targets=target[mask], prediction=True, toPredict=features[not mask])
        print(trainedRidge['Predicted'])
        probaRidge = trainedRidge['Predicted']        
        probaRidge[[i for i in probaRidge if i>0.5]] = 1 
        probaRidge[[i for i in probaRidge if i<0.5]] = 0
        perfArray[idx] = (probaRidge==target)
        mask[idx] = 1 # reset
    perf[alpha] = np.sum(perfArray)

### TODO: compare the perfs in a meaningful way. Right now they're just sums of booleans.

result = open('../results/resultRidge_range32_CVBarthe.csv','w')
result.write("alpha"+","+"avertarget score"+"\n")
listCV = np.empty([len(alphas)], dtype=float)
j = 0


#coefficient, alpha, score, intercept, scoreCV, predictedtargets
print("Start Ridge regression with alphas"+str(alphas))
for i in alphas:
    results = ridgeRegression([i], features, target)
    listCV[j] = results['scoreCV']
    #predictedtargets = results['Predictedtargets']
    # write in a csv file
    result.write(str(i)+","+str(listCV[j])+"\n")
    j = j + 1
result.close()
print(np.amax(listCV))
print(alphas[np.argmax(listCV)])
print("End of computation")
