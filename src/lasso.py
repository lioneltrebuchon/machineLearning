########################################################################

# Python script to output result from Lasso regression

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

# Input (features and age) of the regression
features = np.transpose(np.genfromtxt('../results/sliceFeatures.csv', delimiter="\n"))
age = np.genfromtxt('../data/targets.csv', delimiter="\n")

# Features for the prediction
toPredictFeatures = np.transpose(np.genfromtxt('../results/test_sliceFeatures.csv', delimiter="\n"))

def lassoRegression(alphas, features, age, prediction=False, toPredict=np.empty(1, dtype=int)):
    # More info at :
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html

    # We set up the model
    # cv : cross-validation generator (with none, Generalized Cross-Validation (efficient Leave-One-Out)
    modelLasso = linear_model.LassoCV(alphas, normalize=True, cv=None)

    # We compute the model
    result = modelLasso.fit(features, age)

    # We compute the score
    scoreFinal = modelLasso.score(features, age)

    print("Coefficient: {0} Alpha: {1} Score: {2} Intercept: {3}".format(modelLasso.coef_, modelLasso.alpha_, scoreFinal, modelLasso.intercept_))

    # Prediction
    if prediction==True:
        predictionOutput = modelLasso.predict(toPredict).astype(int)
        print("Prediction: {0}".format(predictionOutput))
        return {'Coefficient': modelLasso.coef_, 'Alpha': modelLasso.alpha_, 'Score': scoreFinal, 'Intercept': modelLasso.intercept_, 'PredictedAges': predictionOutput}
    else:
        return {'Coefficient': modelLasso.coef_, 'Alpha': modelLasso.alpha_, 'Score': scoreFinal, 'Intercept': modelLasso.intercept_}

alphaStart = 0.01
alphaEnd = 1
alphaStep = 0.01

# compute the regression for several alphas
alphas = np.linspace(alphaStart, alphaEnd, (alphaEnd-alphaStart)/alphaStep)
coefficient, alpha, score, intercept, predictedAges = lassoRegression(alphas, features, age, True, toPredictFeatures)

# write in a csv file
fresult = open('../results/resultLasso.csv','w')
fresult.write("ID,Prediction,alpha:,"+alpha+"\n")
for id in range(TEST):
    fresult.write(str(id)+","+str(predictedAges[id])+"\n")
fresult.close()