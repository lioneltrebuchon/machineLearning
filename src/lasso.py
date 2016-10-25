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

fresult = open('../results/resultLasso.csv','w')
fresult.write("ID,Prediction\n")

def lassoRegression(alpha, features, age, prediction=False, toPredict=np.empty(1, dtype=int)):
    # More info at :
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html

    # We set up the model
    # cv : cross-validation generator (with none, Generalized Cross-Validation (efficient Leave-One-Out)
    modelLasso = linear_model.LassoCV(alpha, normalize=True, cv=None)

    # We compute the model
    result = modelLasso.fit(features, age)

    # We compute the score
    scoreFinal = modelLasso.score(features, age)

    print("Coefficient: {0} Alpha: {1} Score: {2} Intercept: {3}".format(modelLasso.coef_, alpha, scoreFinal, modelLasso.intercept_))

    # Prediction
    if prediction==True:
        predictionOutput = modelLasso.predict(toPredict).astype(int)
        print("Prediction: {0}".format(predictionOutput))
        return {'Coefficient': modelLasso.coef_, 'Alpha': alpha, 'Score': scoreFinal, 'Intercept': modelLasso.intercept_, 'PredictedAges': predictionOutput}
    else:
        return {'Coefficient': modelLasso.coef_, 'Alpha': alpha, 'Score': scoreFinal, 'Intercept': modelLasso.intercept_}

predictedAges = lassoRegression([0.1], features, age, True, toPredictFeatures)['PredictedAges']

for id in range(TEST):
    fresult.write(str(id)+","+str(predictedAges[id])+"\n")
fresult.close()