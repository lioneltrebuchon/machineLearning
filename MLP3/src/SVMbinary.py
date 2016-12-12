########################################################################

# Python script to output result from SVM binary classification 

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

# Input (features and target) of the regression
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

# Targets for binary classification
targetGender = np.genfromtxt('../data/targets_gender.csv', delimiter=",")
targetAge = np.genfromtxt('../data/targets_age.csv', delimiter=",")
targetHealth = np.genfromtxt('../data/targets_health.csv', delimiter=",")

# Features for the prediction
# Read features of the test set to predict
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

def svmclassification(features, targets, C=1, kernel='rbf', degree=3, gamma='auto', decision_function_shape=None, prediction=False, toPredict=np.empty(1, dtype=int)):
    # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

    # We set up the model
    modelSVM = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, decision_function_shape=decision_function_shape, probability=True)

    # We compute the model
    modelSVM.fit(features, targets)

    # Compute the distance of the samples X to the separating hyperplane.
    dist = modelSVM.decision_function(features)

    # We compute the score (mean accuracy wrt. to the real output targets)
    # Not very relevant as we compute a score over the training set.
    scoreFinal = modelSVM.score(features, targets)

    #print("SV repartition: {0} Score: {1} Coefficients: {2} Intercept: {3}".format(modelSVM.n_support_, scoreFinal, modelSVM.coef_, modelSVM.intercept_))

    print("SV repartition: {0} Score: {1}".format(modelSVM.n_support_, scoreFinal))

    # Prediction
    if prediction==True:
        predictionOutput = modelSVM.predict(toPredict)
        prob = modelSVM.predict_proba(toPredict)
        return {'SV': modelSVM.support_vectors_, 'SV indices': modelSVM.support_, 'SV repartition': modelSVM.n_support_, 'Coefficients': modelSVM.coef_, 'Intercept': modelSVM.intercept_, 'Score': scoreFinal, 'Predicted': predictionOutput, 'Probabilities': prob}
    else:
        return {'SV': modelSVM.support_vectors_, 'SV indices': modelSVM.support_, 'SV repartition': modelSVM.n_support_, 'Coefficients': modelSVM.coef_, 'Intercept': modelSVM.intercept_, 'Score': scoreFinal}

kernel='linear'

#clist = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
#clist = np.linspace(0.0008, 0.0015, 11)
clist = [0.1]

for c in clist:
#for n in range(-7, 7):
    #c = 10**n

    print("Start SVM classification for gender with C = "+str(c))
    prediction = True
    resultsGender = svmclassification(features, targets=targetGender, C=c, kernel=kernel, degree=2, gamma='auto', decision_function_shape=None, prediction=True, toPredict=toPredictFeatures, probability=False)

    print("Start SVM classification for age with C = "+str(c))
    prediction = True
    resultsAge = svmclassification(features, targets=targetAge, C=c, kernel=kernel, degree=2, gamma='auto', decision_function_shape=None, prediction=True, toPredict=toPredictFeatures, probability=False)

    print("Start SVM classification for health with C = "+str(c))
    prediction = True
    resultsHealth = svmclassification(features, targets=targetHealth, C=c, kernel=kernel, degree=2, gamma='auto', decision_function_shape=None, prediction=True, toPredict=toPredictFeatures, probability=False)

    # Aggregate the 3 binary classifications
    predictedGender = resultsGender['Predicted']
    predictedAge = resultsAge['Predicted']
    predictedHealth = resultsHealth['Predicted']
    probabGender = resultsGender['Probabilities']
    probabAge = resultsAge['Probabilities']
    probabHealth = resultsHealth['Probabilities']
    predictedTransf = [[0 for i in xrange(3)] for i in xrange(TEST)]

    for id in range(TEST):
        if predictedGender[id]==0:
            predictedTransf[id][0] = 'FALSE'
        elif predictedGender[id]==1:
            predictedTransf[id][0] = 'TRUE'
        else:
            predictedTransf[id][0] = 'ERROR'
            print("ERROR for gender for id: "+str(id))

        if predictedAge[id]==0:
            predictedTransf[id][1] = 'FALSE'
        elif predictedAge[id]==1:
            predictedTransf[id][1] = 'TRUE'
        else:
            predictedTransf[id][1] = 'ERROR'
            print("ERROR for age for id: "+str(id))

        if predictedHealth[id]==0:
            predictedTransf[id][2] = 'FALSE'
        elif predictedHealth[id]==1:
            predictedTransf[id][2] = 'TRUE'
        else:
            predictedTransf[id][2] = 'ERROR'
            print("ERROR for health for id: "+str(id))

    # write in a csv file
    result = open('../results/SVMbinary'+kernel+str(c)+'.csv','w')
    result.write("ID,Sample,Label,Predicted"+"\n")
    for id in range(TEST*3):
            if id%3==0:
                #print(predictedTransf[id/3][0])
                result.write(str(id)+","+str(id/3)+",gender,"+predictedTransf[id/3][0]+"\n")
            elif id%3==1:
                #print(predictedTransf[id/3][1])
                result.write(str(id)+","+str(id/3)+",age,"+predictedTransf[id/3][1]+"\n")
            elif id%3==2:
                #print(predictedTransf[id/3][2])
                result.write(str(id)+","+str(id/3)+",health,"+predictedTransf[id/3][2]+"\n")
            else:
                print("Error during prediction for id: "+str(id))
                result.write("ERROR,ERROR,ERROR,ERROR"+"\n")
    result.close()

