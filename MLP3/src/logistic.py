########################################################################

# Python script to output result from Logistic regression

########################################################################

TEST=138
TRAIN=278

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import linear_model
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

# Same targets as for SVM (multi classes)
target = np.genfromtxt('../data/targetsLogistic.csv', delimiter=",")

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

def logisticRegression(features, targets, C=0.1, prediction=False, toPredict=np.empty(1, dtype=int)):
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression

    # We set up the model
    modelLogistic = sk.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=C, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)

    # We compute the model
    result = modelLogistic.fit(features, targets)

    # We compute the score
    scoreFinal = modelLogistic.score(features, targets)

    print("Score: {0} C: {1}".format(scoreFinal, C))

    # Prediction
    if prediction==True:
        predictionOutput = modelLogistic.predict(toPredict)
        prob = modelLogistic.predict_proba(toPredict)
        return {'Coefficient': modelLogistic.coef_, 'C': C, 'Score': scoreFinal, 'Intercept': modelLogistic.intercept_, 'Predicted': predictionOutput, 'Probabilities': prob}
    else:
        return {'Coefficient': modelLogistic.coef_, 'C': C, 'Score': scoreFinal, 'Intercept': modelLogistic.intercept_}

#crange = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
crange = np.linspace(0.00005, 0.0005, 11)
#crange = [6]

for c in crange:
    print("Start Logistic regression with different C: "+str(c))

    prediction = True
    results = logisticRegression(features, targets=target, C=c, prediction=True, toPredict=toPredictFeatures)

    # write in a csv file
    predicted = results['Predicted']
    # Rq: no classes 2 and 6
    probab = results['Probabilities']

    if prediction==True:
        predicted = results['Predicted']
        probab = results['Probabilities']

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
        result = open('../results/logistic'+str(c)+'.csv','w')
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
