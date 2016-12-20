########################################################################
#               Ridge + binary SVM classification                      #
########################################################################

TEST=138
TRAIN=278

C = [0.001]
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
multiTargets = targets[:,0]+2*targets[:,1]+4*targets[:,2]

for c in C:
    ### Multi SVM
    print("Start SVM classification with c = "+str(c))
    SVMResults = svmClassification(featuresScaled, targets=multiTargets, C=c, kernel=kernel, \
    degree=2, gamma='auto', decision_function_shape=None, prediction=True, toPredict=toPredictFeaturesScaled)
    SVMpredictions = SVMResults['Predicted']
    '''SVMprobab = SVMResults['Probabilities']'''

    ### Write in a csv file
    result = open('results/multiSVM_'+str(c)+'.csv','w')
    result.write("ID,Sample,Label,Predicted"+"\n")
    characNames = [',gender,',',age,',',health,']
    for i in range(TEST):
        prediction = SVMpredictions[i]
        for charac in [0,1,2]:
            id = 3*i+charac
            result.write(str(id)+','+str(i)+characNames[charac]+str(bool(prediction%2)).upper()+"\n")
            prediction -= prediction%2
            prediction /= 2
    result.close()
