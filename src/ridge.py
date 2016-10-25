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

fresult = open('../results/resultRidge.csv','w')
fresult.write("ID,Prediction\n")

def ridgeRegression(alpha, features, age, prediction=False, toPredict=np.empty(1, dtype=int)):
    # More info at :
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html

    # We set up the model
    # cv : cross-validation generator (with none, Generalized Cross-Validation (efficient Leave-One-Out)
    modelRidge = linear_model.RidgeCV(alpha, normalize=True, cv=None)

    # We compute the model
    result = modelRidge.fit(features, age)

    # We compute the score
    scoreFinal = modelRidge.score(features, age)

    print("Coefficient: {0} Alpha: {1} Score: {2} Intercept: {3}".format(modelRidge.coef_, alpha, scoreFinal, modelRidge.intercept_))

    # Prediction
    if prediction==True:
        predictionOutput = modelRidge.predict(toPredict).astype(int)
        print("Prediction: {0}".format(predictionOutput))
        return {'Coefficient': modelRidge.coef_, 'Alpha': alpha, 'Score': scoreFinal, 'Intercept': modelRidge.intercept_, 'PredictedAges': predictionOutput}
    else:
        return {'Coefficient': modelRidge.coef_, 'Alpha': alpha, 'Score': scoreFinal, 'Intercept': modelRidge.intercept_}

# # Compute the reg for several alpha and determin the prediction for the best alpha
# alphaStart = 0.1
# alphaENd = 1
# alphaStep = 0.1
#
# #for i in np.linspace(alphaStart, alphaENd, 100):
# for i in range(alphaStart, alphaENd, alphaStep):
#     result = ridgeRegression([i], features, age)
#     RidgeAlpha.append(i)
#     RidgeScore.append(result['Score'])
#
# plt.plot(RidgeAlpha, RidgeScore)
# plt.xlabel('Alpha')
# plt.ylabel('Score')
# plt.show()
#
# maxScore = amax(RidgeScore, axis=0)
# maxAlpha = RidgeAlpha[RidgeScore.argmax(axis=0)]

predictedAges = ridgeRegression([0.1], features, age, True, toPredictFeatures)['PredictedAges']

for id in range(TEST):
    fresult.write(str(id)+","+str(predictedAges[id])+"\n")
fresult.close()