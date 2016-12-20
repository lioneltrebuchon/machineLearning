########################################################################

# Python script to output result from SVM classification

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

### Input (features and target) of the regression ###
sectionFeatures = np.genfromtxt('../features/train_section_features.csv', delimiter=",")
segmentFeatures = np.genfromtxt('../features/train_segment_features.csv', delimiter=",")

features = preprocessing.scale(np.concatenate([sectionFeatures, segmentFeatures],1))

# Features for the prediction
# Read features of the test set to predict

sectionFeatures = np.genfromtxt('../features/test_section_features.csv', delimiter=",")
segmentFeatures = np.genfromtxt('../features/test_segment_features.csv', delimiter=",")

toPredictFeatures = preprocessing.scale(np.concatenate([sectionFeatures,segmentFeatures],1))

# Targets for Ridge and SVM multi labels
targets = np.genfromtxt('../data/targets.csv', delimiter=",")

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

#clist = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
#clist = np.linspace(0.0008, 0.001, 11)
clists = [[0.0035,0.0014,0.0002]]
kernel='linear'

predicted = [[],[],[]]
probab = [[],[],[]]

for clist in clists:
    for charac in [0,1,2]:
    #for n in range(-7, 7):
        #c = 10**n
        print("Start SVM classification of characteristic = "+str(charac)+" with c = "+str(clist[charac]))

        prediction = True
        #results = svmclassification(features, targets, C=c, kernel=kernel, degree=2, gamma='auto', decision_function_shape=None, prediction=False, toPredict=np.empty(1, dtype=int))
        results = svmclassification(features, targets=targets[:,charac], C=clist[charac], kernel=kernel, degree=2, gamma='auto', decision_function_shape=None, prediction=True, toPredict=toPredictFeatures)

        # write in a csv file
        if prediction==True:
            predicted[charac] = results['Predicted']
            probab[charac] = results['Probabilities']
            print(predicted[charac])
            print(probab[charac])

    # write in a csv file
    result = open('../results/SVM3'+kernel+str(clist)+'.csv','w')
    result.write("ID,Sample,Label,Predicted"+"\n")
    for id in range(TEST*3):
            if id%3==0:
                result.write(str(id)+","+str(id/3)+",gender,"+str(bool(predicted[0][id/3])).upper()+"\n")
            elif id%3==1:
                result.write(str(id)+","+str(id/3)+",age,"+str(bool(predicted[1][id/3])).upper()+"\n")
            elif id%3==2:
                result.write(str(id)+","+str(id/3)+",health,"+str(bool(predicted[2][id/3])).upper()+"\n")
            else:
                print("Error during prediction for id: "+str(id))
                result.write("ERROR,ERROR,ERROR,ERROR"+"\n")
    result.close()
