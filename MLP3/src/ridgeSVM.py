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
p2x = np.genfromtxt('../features/train_p2_x.csv', delimiter=",")
p2y = np.genfromtxt('../features/train_p2_y.csv', delimiter=",")
p3x = np.genfromtxt('../features/train_p3_x.csv', delimiter=",")
p3y = np.genfromtxt('../features/train_p3_y.csv', delimiter=",")
p2x = preprocessing.scale(p2x)
p2y = preprocessing.scale(p2y)
p3x = preprocessing.scale(p3x)
p3y = preprocessing.scale(p3y)

sectionFeatures = np.genfromtxt('../features/train_section_features.csv', delimiter=",")
sectionFeatures = preprocessing.scale(sectionFeatures)

features = np.concatenate([np.reshape(p2x,[-1,1]),np.reshape(p2y,[-1,1]),np.reshape(p3x,[-1,1]),np.reshape(p3y,[-1,1]),sectionFeatures],1)

# Targets for Ridge and SVM multi labels
targetSVM = np.genfromtxt('../data/targetsSVM.csv', delimiter=",")
targetRidge = np.genfromtxt('../data/targets.csv', delimiter=",")

### Features for the prediction ###
p2x = np.genfromtxt('../features/test_p2_x.csv', delimiter=",")
p2y = np.genfromtxt('../features/test_p2_y.csv', delimiter=",")
p3x = np.genfromtxt('../features/test_p3_x.csv', delimiter=",")
p3y = np.genfromtxt('../features/test_p3_y.csv', delimiter=",")
p2x = preprocessing.scale(p2x)
p2y = preprocessing.scale(p2y)
p3x = preprocessing.scale(p3x)
p3y = preprocessing.scale(p3y)

sectionFeatures = np.genfromtxt('../features/test_section_features.csv', delimiter=",")
sectionFeatures = preprocessing.scale(sectionFeatures)

toPredictFeatures = np.concatenate([np.reshape(p2x,[-1,1]),np.reshape(p2y,[-1,1]),np.reshape(p3x,[-1,1]),np.reshape(p3y,[-1,1]),sectionFeatures],1)


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

    print("SV repartition: {0} Score: {1}".format(modelSVM.n_support_, scoreFinal))

    # Prediction
    if prediction==True:
        predictionOutput = modelSVM.predict(toPredict)
        prob = modelSVM.predict_proba(toPredict)
        return {'SV': modelSVM.support_vectors_, 'SV indices': modelSVM.support_, 'SV repartition': modelSVM.n_support_, 'Coefficients': modelSVM.coef_, 'Intercept': modelSVM.intercept_, 'Score': scoreFinal, 'Predicted': predictionOutput, 'Probabilities': prob}
    else:
        return {'SV': modelSVM.support_vectors_, 'SV indices': modelSVM.support_, 'SV repartition': modelSVM.n_support_, 'Coefficients': modelSVM.coef_, 'Intercept': modelSVM.intercept_, 'Score': scoreFinal}


### Ridge regression classification ###
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

print("Start SVM (C= "+str(c)+", kernel: "+kernel+") and ridge classification (alpha= "+str(alpha)+")")

resultsSVM = svmclassification(features, targets=targetSVM, C=c, kernel=kernel, degree=2, gamma='auto', decision_function_shape=None, prediction=True, toPredict=toPredictFeatures)
resultsRidge = ridgeRegression(alphas=[alpha], features=features, targets=targetRidge, prediction=True, toPredict=toPredictFeatures)

### Transform ridge and SVM so that they can be weighted and summed ###

#Ridge

probaRidge = resultsRidge['Predicted']
print("Ridge normalized first line:  "+str(probaRidge[0,:]))
probaRidge = preprocessing.scale(probaRidge) #normalize  TODO NOT WORKING
print("Ridge normalized first line:  "+str(probaRidge[0,:]))

#SVM

probaSVM = resultsSVM['Probabilities']
# We need to add the missing classes 2 and 6
probaSVMFull = np.insert(probaSVM, 1, 0, axis =1)
probaSVMFull = np.insert(probaSVMFull, 5, 0, axis =1)

# We transform the class probabilities of SVM for 8 classes into a subset of 3 classes thanks to the transformation matrix matrixOfTransformation
column1 =  np.transpose(np.array([1, 1, 1, 1, 0, 0, 0, 0]))
column2 =  np.transpose(np.array([1, 1, 0, 0, 1, 1, 0, 0]))
column3 =  np.transpose(np.array([1, 0, 1, 0, 1, 0, 1, 0]))
matrixOfTransformation =  [[0 for i in xrange(3)] for i in xrange(8)]
matrixOfTransformation = np.concatenate([np.reshape(column1,[-1,1]),np.reshape(column2,[-1,1]),np.reshape(column3,[-1,1])],1)

probaSVMTransformed =  [[0 for i in xrange(3)] for i in xrange(TEST)]
probaSVMTransformed = np.dot(probaSVMFull, matrixOfTransformation)
print("SVM first line:  "+str(probaSVMTransformed[0,:]))

### Weighted sum of ridge and SVM ###

weightedProba = [[0 for i in xrange(3)] for i in xrange(TEST)]
weightedProba = 0.5*probaSVMTransformed + 0.5*probaRidge
print("Final weighted first line:  "+str(weightedProba[0,:]))
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
result = open('../results/ridgeSVM'+str(c)+str(alpha)+'.csv','w')
result.write("ID,Sample,Label,Predicted"+"\n")
for id in range(TEST*3):
    if id%3==0:
        result.write(str(id)+","+str(id/3)+",gender,"+weightedProbaRounded[id/3][0]+"\n")
    elif id%3==1:
        result.write(str(id)+","+str(id/3)+",age,"+weightedProbaRounded[id/3][1]+"\n")
    elif id%3==2:
        result.write(str(id)+","+str(id/3)+",health,"+weightedProbaRounded[id/3][2]+"\n")
    else:
        print("Error during prediction for id: "+str(id))
        result.write("ERROR,ERROR,ERROR,ERROR"+"\n")
result.close()
