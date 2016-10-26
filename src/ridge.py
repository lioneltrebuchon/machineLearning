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

    print("Coefficient: {0} Alpha: {1} Score: {2} Intercept: {3}".format(modelRidge.coef_, modelRidge.alpha_, scoreFinal, modelRidge.intercept_))

    # Prediction
    if prediction==True:
        predictionOutput = modelRidge.predict(toPredict).astype(int)
        print("Prediction: {0}".format(predictionOutput))
        return {'Coefficient': modelRidge.coef_, 'Alpha': modelRidge.alpha_, 'Score': scoreFinal, 'Intercept': modelRidge.intercept_, 'PredictedAges': predictionOutput}
    else:
        return {'Coefficient': modelRidge.coef_, 'Alpha': modelRidge.alpha_, 'Score': scoreFinal, 'Intercept': modelRidge.intercept_}

alphaStart = 0.01
alphaEnd = 10
alphaStep = 0.01

# compute the regression for several alphas
alphas = np.linspace(alphaStart, alphaEnd, (alphaEnd-alphaStart)/alphaStep)
coefficient, alpha, score, intercept, predictedAges = ridgeRegression(alphas, features, age, True, toPredictFeatures)

# write in a csv file
fresult = open('../results/resultRidge.csv','w')
fresult.write("ID,Prediction,alpha:,"+alpha+"\n")
for id in range(TEST):
    fresult.write(str(id)+","+str(predictedAges[id])+"\n")
fresult.close()