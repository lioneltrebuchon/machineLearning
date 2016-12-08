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

target = np.genfromtxt('../data/targetsSVM.csv', delimiter=",")

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
    modelSVM = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, decision_function_shape=decision_function_shape)

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
        predictionOutput = (modelSVM.predict(toPredict))
        return {'SV': modelSVM.support_vectors_, 'SV indices': modelSVM.support_, 'SV repartition': modelSVM.n_support_, 'Coefficients': modelSVM.coef_, 'Intercept': modelSVM.intercept_, 'Score': scoreFinal, 'Predicted': predictionOutput}
    else:
        return {'SV': modelSVM.support_vectors_, 'SV indices': modelSVM.support_, 'SV repartition': modelSVM.n_support_, 'Coefficients': modelSVM.coef_, 'Intercept': modelSVM.intercept_, 'Score': scoreFinal}

#clist = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
#clist = np.linspace(0.0008, 0.001, 11)
clist = [0.001]
kernel='linear'

for c in clist:
#for n in range(-7, 7):
    #c = 10**n
    print("Start SVM classification with C = "+str(c))

    prediction = True
    #results = svmclassification(features, targets, C=c, kernel=kernel, degree=2, gamma='auto', decision_function_shape=None, prediction=False, toPredict=np.empty(1, dtype=int))
    results = svmclassification(features, targets=target, C=c, kernel=kernel, degree=2, gamma='auto', decision_function_shape=None, prediction=True, toPredict=toPredictFeatures)

    # write in a csv file
    if prediction==True:
        predicted = results['Predicted']

        # Transform the 7 multi-classes in 3 binary subclasses
        predictedTransf = [[0 for i in xrange(3)] for i in xrange(TEST)]
        for id in range(TEST):
            if predicted[id]==0:
                predictedTransf[id][0] = 'FALSE'
                predictedTransf[id][1] = 'FALSE'
                predictedTransf[id][2] = 'FALSE'
            elif predicted[id]==1:
                predictedTransf[id][0] = 'FALSE'
                predictedTransf[id][1] = 'FALSE'
                predictedTransf[id][2] = 'TRUE'
            elif predicted[id]==2:
                predictedTransf[id][0] = 'FALSE'
                predictedTransf[id][1] = 'TRUE'
                predictedTransf[id][2] = 'FALSE'
            elif predicted[id]==3:
                predictedTransf[id][0] = 'FALSE'
                predictedTransf[id][1] = 'TRUE'
                predictedTransf[id][2] = 'TRUE'
            elif predicted[id]==4:
                predictedTransf[id][0] = 'TRUE'
                predictedTransf[id][1] = 'FALSE'
                predictedTransf[id][2] = 'FALSE'
            elif predicted[id]==5:
                predictedTransf[id][0] = 'TRUE'
                predictedTransf[id][1] = 'FALSE'
                predictedTransf[id][2] = 'TRUE'
            elif predicted[id]==6:
                predictedTransf[id][0] = 'TRUE'
                predictedTransf[id][1] = 'TRUE'
                predictedTransf[id][2] = 'FALSE'
            elif predicted[id]==7:
                predictedTransf[id][0] = 'TRUE'
                predictedTransf[id][1] = 'TRUE'
                predictedTransf[id][2] = 'TRUE'
            else:
                predictedTransf[id][0] = 'ERROR'
                predictedTransf[id][1] = 'ERROR'
                predictedTransf[id][2] = 'ERROR'
                print("ERROR for id: "+str(id))

        # write in a csv file
        result = open('../results/SVM'+kernel+str(c)+'.csv','w')
        result.write("ID,Sample,Label,Predicted"+"\n")
        for id in range(TEST*3):
                if id%3==0:
                    result.write(str(id)+","+str(id/3)+",gender,"+predictedTransf[id/3][0]+"\n")
                elif id%3==1:
                    result.write(str(id)+","+str(id/3)+",age,"+predictedTransf[id/3][1]+"\n")
                elif id%3==2:
                    result.write(str(id)+","+str(id/3)+",health,"+predictedTransf[id/3][2]+"\n")
                else:
                    print("Error during prediction for id: "+str(id))
                    result.write("ERROR,ERROR,ERROR,ERROR"+"\n")
        result.close()

