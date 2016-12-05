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


# Input (features and "age" (diseased or nor)) of the regression
p2x = np.genfromtxt('../features/train_p2_x.csv', delimiter=",")
p2y = np.genfromtxt('../features/train_p2_y.csv', delimiter=",")
p3x = np.genfromtxt('../features/train_p3_x.csv', delimiter=",")
p3y = np.genfromtxt('../features/train_p3_y.csv', delimiter=",")
sectionFeatures = np.genfromtxt('../features/train_section_features.csv', delimiter=",")
features = np.concatenate([np.reshape(p2x,[-1,1]),np.reshape(p2y,[-1,1]),np.reshape(p3x,[-1,1]),np.reshape(p3y,[-1,1]),sectionFeatures],1)

age = np.genfromtxt('../data/targets.csv', delimiter="\n")

# Features for the prediction
# Read features of the test set to predict
p2x = np.genfromtxt('../features/test_p2_x.csv', delimiter=",")
p2y = np.genfromtxt('../features/test_p2_y.csv', delimiter=",")
p3x = np.genfromtxt('../features/test_p3_x.csv', delimiter=",")
p3y = np.genfromtxt('../features/test_p3_y.csv', delimiter=",")
sectionFeatures = np.genfromtxt('../features/test_section_features.csv', delimiter=",")
toPredictFeatures = np.concatenate([np.reshape(p2x,[-1,1]),np.reshape(p2y,[-1,1]),np.reshape(p3x,[-1,1]),np.reshape(p3y,[-1,1]),sectionFeatures],1)

def ridgeRegression(alphas, features, age, prediction=False, toPredict=np.empty(1, dtype=int)):
    # More info at :
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html

    # We set up the model
    # cv : cross-validation generator (with none, Generalized Cross-Validation (efficient Leave-One-Out)
    modelRidge = linear_model.RidgeCV(alphas, normalize=True, cv=None)

    # We compute the model
    result = modelRidge.fit(features, age)

    # We compute the score
    scoreFinal = modelRidge.score(features, age)

    #print("Coefficient: {0} Alpha: {1} Score: {2} Intercept: {3}".format(modelRidge.coef_, modelRidge.alpha_, scoreFinal, modelRidge.intercept_))

    # Prediction
    if prediction==True:
        predictionOutput = modelRidge.predict(toPredict)
        return {'Coefficient': modelRidge.coef_, 'Alpha': modelRidge.alpha_, 'Score': scoreFinal, 'Intercept': modelRidge.intercept_, 'PredictedAges': predictionOutput}
    else:
        return {'Coefficient': modelRidge.coef_, 'Alpha': modelRidge.alpha_, 'Score': scoreFinal, 'Intercept': modelRidge.intercept_}

# compute the regression for several alphas
#alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
alphas = np.linspace(5, 7, 20)
#alphas = [6]

for alpha in alphas:
    #coefficient, alpha, score, intercept, predictedAges
    print("Start Ridge regression with different alphas "+str(alpha))
    results = ridgeRegression([alpha], features, age, True, toPredict=toPredictFeatures)
    #print("End of ridge "+str(alpha))
    predictedAges = results['PredictedAges']

    #print(predictedAges)

    # write in a csv file
    result = open('../results/ridgeTestPrepro'+str(alpha)+'.csv','w')
    result.write("ID,Prediction"+"\n")
    testIs2or3 = np.genfromtxt('../data/testIs2or3.csv', delimiter="\n")
    for id in range(TEST):
        if testIs2or3[id]==2 or predictedAges[id]>1:
            result.write(str(id+1)+","+str(1)+"\n")
        elif predictedAges[id]<0:
            result.write(str(id+1)+","+str(0)+"\n")
        else:
            result.write(str(id+1)+","+str(predictedAges[id])+"\n")
            #print(str(predictedAges[id]))
    result.close()
