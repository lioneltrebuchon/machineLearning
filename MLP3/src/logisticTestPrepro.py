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
sectionFeatures = np.genfromtxt('../features/train_section_features.csv', delimiter=",")
features = np.concatenate([np.reshape(p2x,[-1,1]),np.reshape(p2y,[-1,1]),np.reshape(p3x,[-1,1]),np.reshape(p3y,[-1,1]),sectionFeatures],1)

target = np.genfromtxt('../data/targets.csv', delimiter="\n")

# Features for the prediction
# Read features of the test set to predict
p2x = np.genfromtxt('../features/test_p2_x.csv', delimiter=",")
p2y = np.genfromtxt('../features/test_p2_y.csv', delimiter=",")
p3x = np.genfromtxt('../features/test_p3_x.csv', delimiter=",")
p3y = np.genfromtxt('../features/test_p3_y.csv', delimiter=",")
sectionFeatures = np.genfromtxt('../features/test_section_features.csv', delimiter=",")
toPredictFeatures = np.concatenate([np.reshape(p2x,[-1,1]),np.reshape(p2y,[-1,1]),np.reshape(p3x,[-1,1]),np.reshape(p3y,[-1,1]),sectionFeatures],1)

def logisticRegression(features, target, c=0.1, prediction=False, toPredict=np.empty(1, dtype=int)):
    # More info at :
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict

    # We set up the model
    modelLogistic = sk.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=c, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)

    # We compute the model
    result = modelLogistic.fit(features, target)

    # We compute the score
    scoreFinal = modelLogistic.score(features, target)

    print("Coefficient: {0} C: {1} Score: {2} Intercept: {3}".format(modelLogistic.coef_, c, scoreFinal, modelLogistic.intercept_))

    # Prediction
    if prediction==True:
        predictionOutput = modelLogistic.predict(toPredict)
        return {'Coefficient': modelLogistic.coef_, 'C': c, 'Score': scoreFinal, 'Intercept': modelLogistic.intercept_, 'Predicted': predictionOutput}
    else:
        return {'Coefficient': modelLogistic.coef_, 'C': c, 'Score': scoreFinal, 'Intercept': modelLogistic.intercept_}

# compute the regression for several C
#crange = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
#crange = np.linspace(1, 10, 10)
crange = [6]

for c in crange:
    print("Start Logistic regression with different C "+str(c))
    results = logisticRegression(features, target, c, True, toPredict=toPredictFeatures)
    predicted = results['Predicted']

    print(predicted)

################# LOGISTIC NOT UPDATED HERE

    # write in a csv file
    result = open('../results/logisticTestPrepro'+str(c)+'.csv','w')
    result.write("ID,Prediction"+"\n")
    testIs2or3 = np.genfromtxt('../data/testIs2or3.csv', delimiter="\n")
    for id in range(TEST):
        if testIs2or3[id]==2 or predicted[id]>1:
            result.write(str(id+1)+","+str(1)+"\n")
        elif predicted[id]<0:
            result.write(str(id+1)+","+str(0)+"\n")
        else:
            result.write(str(id+1)+","+str(predicted[id])+"\n")
            #print(str(predicted[id]))
    result.close()
