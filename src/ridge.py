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

# feature1 = np.genfromtxt('../results/p2_x.csv', delimiter="\n")
# feature2 = np.genfromtxt('../results/p3_x.csv', delimiter="\n")
# feature3 = np.genfromtxt('../results/p2_y.csv', delimiter="\n")
# feature4 = np.genfromtxt('../results/p3_y.csv', delimiter="\n")
#
# features5 = np.transpose(np.array([feature1, feature2, feature3, feature4]))

# Input (features and age) of the regression
features = np.genfromtxt('../results/sliceFeatures.csv', delimiter=",")
age = np.genfromtxt('../data/targets.csv', delimiter="\n")

# print(features5)
# print(features5.shape)
# print(age)
# print(age.shape)
print(features)
print(features.shape)

'''
# Features for the prediction
# toPredictFeatures = np.transpose(np.genfromtxt('../results/test_sliceFeatures.csv', delimiter="\n"))

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

# alphaStart = 0.1
# alphaEnd = 1
# alphaStep = 0.1

# compute the regression for several alphas
# alphas = np.linspace(alphaStart, alphaEnd, (alphaEnd-alphaStart)/alphaStep)
alphas = [0.1]
# coefficient, alpha, score, intercept, predictedAges = ridgeRegression(alphas, features, age, True, toPredictFeatures)
coefficient, alpha, score, intercept = ridgeRegression(alphas, features, age)

# # write in a csv file
# fresult = open('../results/resultRidge.csv','w')
# fresult.write("ID,Prediction,alpha:,"+alpha+"\n")
# for id in range(TEST):
#     fresult.write(str(id)+","+str(predictedAges[id])+"\n")
# fresult.close()
'''