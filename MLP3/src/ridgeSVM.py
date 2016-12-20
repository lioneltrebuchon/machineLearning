########################################################################

# Output result from a combined Ridge and SVM classification

########################################################################

TEST=138
TRAIN=278

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import svm
from sklearn import preprocessing
from sklearn import linear_model

### Input (features and target) of the regression ###
sectionFeatures = np.genfromtxt('../features/train_section_features.csv', delimiter=",")
segmentFeatures = np.genfromtxt('../features/train_segment_features.csv', delimiter=",")

features = np.concatenate([sectionFeatures, segmentFeatures],1)

# Features for the prediction
# Read features of the test set to predict

sectionFeatures = np.genfromtxt('../features/test_section_features.csv', delimiter=",")
segmentFeatures = np.genfromtxt('../features/test_segment_features.csv', delimiter=",")

toPredictFeatures = np.concatenate([sectionFeatures,segmentFeatures],1)

# Targets for Ridge and SVM multi labels
targetSVM = np.genfromtxt('../data/targetsSVM.csv', delimiter=",")
targetRidge = np.genfromtxt('../data/targets.csv', delimiter=",")

### SVM classification ###
def svmclassification(features, targets, C=1, kernel='rbf', degree=3, gamma='auto', decision_function_shape=None, prediction=False, toPredict=np.empty(1, dtype=int)):

    # We set up the model
    modelSVM = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, decision_function_shape=decision_function_shape, probability=True)

    # We compute the model
    modelSVM.fit(features, targets)

    # Compute the distance of the samples X to the separating hyperplane.
    dist = modelSVM.decision_function(features)

    # We compute the score (mean accuracy wrt. to the real output targets)
    # Not very relevant as we compute a score over the training set.
    scoreFinal = modelSVM.score(features, targets)

    #print("SV repartition: {0} Score: {1}".format(modelSVM.n_support_, scoreFinal))

    # Prediction
    if prediction==True:
        predictionOutput = modelSVM.predict(toPredict)
        prob = modelSVM.predict_proba(toPredict)
        return {'SV': modelSVM.support_vectors_, 'SV indices': modelSVM.support_, 'SV repartition': modelSVM.n_support_, 'Coefficients': modelSVM.coef_, 'Intercept': modelSVM.intercept_, 'Score': scoreFinal, 'Predicted': predictionOutput, 'Probabilities': prob}
    else:
        return {'SV': modelSVM.support_vectors_, 'SV indices': modelSVM.support_, 'SV repartition': modelSVM.n_support_, 'Coefficients': modelSVM.coef_, 'Intercept': modelSVM.intercept_, 'Score': scoreFinal}


### Ridge regression ###
def ridgeRegression(alphas, features, targets, prediction=False, toPredict=np.empty(1, dtype=int)):
    modelRidge = linear_model.RidgeCV(alphas, normalize=True, cv=None)

    # We compute the model
    result = modelRidge.fit(features, targets)

    # We compute the score
    scoreFinal = modelRidge.score(features, targets)

    #print("Coefficient: {0} Alpha: {1} Score: {2} Intercept: {3}".format(modelRidge.coef_, modelRidge.alpha_, scoreFinal, modelRidge.intercept_))

    # Prediction
    if prediction==True:
        predictionOutput = modelRidge.predict(toPredict)
        return {'Coefficient': modelRidge.coef_, 'Alpha': modelRidge.alpha_, 'Score': scoreFinal, 'Intercept': modelRidge.intercept_, 'Predicted': predictionOutput}
    else:
        return {'Coefficient': modelRidge.coef_, 'Alpha': modelRidge.alpha_, 'Score': scoreFinal, 'Intercept': modelRidge.intercept_}


### Set parameters for SVM and Ridge. ###
alphas = 0.5  # ridge
clist = 0.001 # SVM
kernel='linear'
c = clist
alpha = alphas

print("Start SVM (C="+str(c)+", kernel:"+kernel+") and Ridge regression (alpha="+str(alpha)+")")

resultsSVM = svmclassification(features, targets=targetSVM, C=c, kernel=kernel, degree=2, gamma='auto', decision_function_shape=None, prediction=True, toPredict=toPredictFeatures)
resultsRidge = ridgeRegression(alphas=[alpha], features=features, targets=targetRidge, prediction=True, toPredict=toPredictFeatures)

### Transform ridge and SVM so that they can be weighted and summed ###

probaRidge = resultsRidge['Predicted']

# We round up Ridge predictions probabilities
probaRidgeTransformed = [[0 for i in xrange(3)] for i in xrange(TEST)]

for id in range(TEST):

    if probaRidge[id][0]<0:
        probaRidgeTransformed[id][0] = 0
    elif probaRidge[id][0]>1:
        probaRidgeTransformed[id][0] = 1
    else:
        probaRidgeTransformed[id][0] = probaRidge[id][0]

    if probaRidge[id][1]<0:
        probaRidgeTransformed[id][1] = 0
    elif probaRidge[id][1]>1:
        probaRidgeTransformed[id][1] = 1
    else:
        probaRidgeTransformed[id][1] = probaRidge[id][1]

    if probaRidge[id][2]<0:
        probaRidgeTransformed[id][2] = 0
    elif probaRidge[id][2]>1:
        probaRidgeTransformed[id][2] = 1
    else:
        probaRidgeTransformed[id][2] = probaRidge[id][2]

probaRidgeTransformed = np.array(probaRidgeTransformed)

'''
print(probaRidge)
print(probaRidge.shape)
print(probaRidgeTransformed)
print(probaRidgeTransformed.shape)
'''

probaSVM = resultsSVM['Probabilities']

'''
print(probaRidge.shape[0])
print("Ridge first line:  "+str(probaRidge[0,:]))
probaRidge[probaRidge<0] = 0
probaRidge[probaRidge>1] = 1
# for i in range(probaRidge.shape[0]):
# 	difference = max(probaRidge[i,:])-min(probaRidge[i,:])
# 	probaRidge[i,:] = (probaRidge[i,:]-min(probaRidge[i,:])) / difference  # normalize
# print("Ridge normalized first line:  "+str(probaRidge[0,:]))
probaSVM = 1-resultsSVM['Probabilities']
'''

# We need to add the missing classes 2 and 6
probaSVMFull = np.insert(probaSVM, 1, 0, axis =1)
probaSVMFull = np.insert(probaSVMFull, 5, 0, axis =1)

# We transform the class probabilities of SVM for 8 classes into a subset of 3 classes thanks to the transformation matrix matrixOfTransformation
column1 =  np.transpose(np.array([0, 0, 0, 0, 1, 1, 1, 1]))
column2 =  np.transpose(np.array([0, 0, 1, 1, 0, 0, 1, 1]))
column3 =  np.transpose(np.array([0, 1, 0, 1, 0, 1, 0, 1]))
matrixOfTransformation =  [[0 for i in xrange(3)] for i in xrange(8)]
matrixOfTransformation = np.concatenate([np.reshape(column1,[-1,1]),np.reshape(column2,[-1,1]),np.reshape(column3,[-1,1])],1)

probaSVMTransformed =  [[0 for i in xrange(3)] for i in xrange(TEST)]
probaSVMTransformed = np.dot(probaSVMFull, matrixOfTransformation)
probaSVMTransformed = np.array(probaSVMTransformed)
print(probaSVMTransformed)
print(probaSVMTransformed.shape)

'''
# We check that the probabilities are correct: we compute the complementary probabilities and by summing, we should always obtain 1
column12 =  np.transpose(np.array([0, 0, 0, 0, 1, 1, 1, 1]))
column22 =  np.transpose(np.array([0, 0, 1, 1, 0, 0, 1, 1]))
column32 =  np.transpose(np.array([0, 1, 0, 1, 0, 1, 0, 1]))
matrixOfTransformation2 =  [[0 for i in xrange(3)] for i in xrange(8)]
matrixOfTransformation2 = np.concatenate([np.reshape(column12,[-1,1]),np.reshape(column22,[-1,1]),np.reshape(column32,[-1,1])],1)

probaSVMTransformed2 =  [[0 for i in xrange(3)] for i in xrange(TEST)]
probaSVMTransformed2 = np.dot(probaSVMFull, matrixOfTransformation2)
probaSVMTransformed2 = np.array(probaSVMTransformed2)

test = np.array([[0 for i in xrange(3)] for i in xrange(TEST)])
test = probaSVMTransformed + probaSVMTransformed2

print(test)
'''

### Weighted sum of ridge and SVM ###

weightedProba = np.array([[0 for i in xrange(3)] for i in xrange(TEST)])
weightedProba = 1*probaSVMTransformed + 0*probaRidgeTransformed
#print("Final weighted first line:  "+str(weightedProba[0,:]))
weightedProbaRounded = [[0 for i in xrange(3)] for i in xrange(TEST)]

# Round up predictions
for id in range(TEST):
    weightedProbaRounded[id][0] = round(weightedProba[id][0])
    if weightedProbaRounded[id][0]<=0:
        weightedProbaRounded[id][0] = 'FALSE'
    else:
        weightedProbaRounded[id][0] = 'TRUE'

    weightedProbaRounded[id][1] = round(weightedProba[id][1])
    if weightedProbaRounded[id][1]<=0:
        weightedProbaRounded[id][1] = 'FALSE'
    else:
        weightedProbaRounded[id][1] = 'TRUE'

    weightedProbaRounded[id][2] = round(weightedProba[id][2])
    if weightedProbaRounded[id][2]<=0:
        weightedProbaRounded[id][2] = 'FALSE'
    else:
        weightedProbaRounded[id][2] = 'TRUE'

# write in a csv file
result = open('../results/ridgeSVMc'+str(c)+'alpha'+str(alpha)+'.csv','w')
result.write("ID,Sample,Label,Predicted"+"\n")
for id in range(TEST*3):
    if id%3==0:
        result.write(str(id)+","+str(id/3)+",gender,"+weightedProbaRounded[id/3][0]+'\n')
    elif id%3==1:
        result.write(str(id)+","+str(id/3)+",age,"+weightedProbaRounded[id/3][1]+'\n')
    elif id%3==2:
        result.write(str(id)+","+str(id/3)+",health,"+weightedProbaRounded[id/3][2]+'\n')
    else:
        print("Error during prediction for id: "+str(id))
        result.write("ERROR,ERROR,ERROR,ERROR"+"\n")
result.close()
