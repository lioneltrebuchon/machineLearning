########################################################################

# Python script to output result from linear regression

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

# Prepare the different features for the regression and then create the input array features
feature1 = np.genfromtxt('../results/p2_x.csv', delimiter="\n")
feature2 = np.genfromtxt('../results/p3_x.csv', delimiter="\n")
feature3 = np.genfromtxt('../results/p2_y.csv', delimiter="\n")
feature4 = np.genfromtxt('../results/p3_y.csv', delimiter="\n")

features = np.transpose(np.array([feature1, feature2, feature3, feature4]))

age = np.genfromtxt('../data/targets.csv', delimiter="\n")

toPredictFeature1 = np.genfromtxt('../results/test_p2_x.csv', delimiter="\n")
toPredictFeature2 = np.genfromtxt('../results/test_p3_x.csv', delimiter="\n")
toPredictFeature3 = np.genfromtxt('../results/test_p2_y.csv', delimiter="\n")
toPredictFeature4 = np.genfromtxt('../results/test_p3_y.csv', delimiter="\n")

toPredictFeatures = np.transpose(np.array([toPredictFeature1, toPredictFeature2, toPredictFeature3, toPredictFeature4]))

fresult = open('../results/result.csv','w')
fresult.write("ID,Prediction\n")

def linearRegression(features, age, prediction=False, toPredict=np.empty(1, dtype=int)):
    # More info at :
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

    # We set up the model
    modelLinear = linear_model.LinearRegression(normalize=True)

    # We compute the model
    result = modelLinear.fit(features, age)

    # We compute the score
    finalScore = modelLinear.score(features, age)

    print("Coefficient: {0} Score: {1} Intercept: {2} Residue: {3}".format(modelLinear.coef_, finalScore, modelLinear.intercept_, modelLinear.residues_, ))

    # Prediction
    if prediction==True:
        predictionOutput = modelLinear.predict(toPredict).astype(int)
        print("Prediction: {0}".format(predictionOutput))
        return {'Coefficient': modelLinear.coef_, 'Score': finalScore, 'Intercept': modelLinear.intercept_, 'Residue': modelLinear.residues_, 'PredictedAges': predictionOutput}
    else:
        return {'Coefficient': modelLinear.coef_, 'Score': finalScore, 'Intercept': modelLinear.intercept_, 'Residue': modelLinear.residues_}


predictedAges=linearRegression(features, age, True, toPredictFeatures)['PredictedAges']

for id in range(TEST):
    fresult.write(str(id)+","+str(predictedAges[id])+"\n")
fresult.close()

