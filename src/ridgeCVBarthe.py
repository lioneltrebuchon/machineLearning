########################################################################

# Python script to output result from Ridge regression

########################################################################

TEST=92
TRAIN=186

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import linear_model
from sklearn.linear_model import Lasso

# Input (features and age) of the regression
features = np.genfromtxt('../results/sliceTrainFeatures32CVBarthe.csv', delimiter=",")
age = np.genfromtxt('../data/targetsTrainCVBarthe.csv', delimiter="\n")

# Features for the prediction
toPredictFeatures = np.genfromtxt('../results/sliceTrainFeatures32ValidationCVBarthe.csv', delimiter=",")

print(features.shape)
print(age.shape)
print(toPredictFeatures.shape)

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

# alphaStart = 0.01
# alphaEnd = 10
# alphaStep = 0.01

# compute the regression for several alphas
alphas = [0.001,0.01,0.1,0.5,1,5,10,50,100]

#coefficient, alpha, score, intercept, predictedAges 
results = ridgeRegression(alphas, features, age, True, toPredict=toPredictFeatures)
alpha = results['Alpha']
predictedAges = results['PredictedAges']

print(alpha)
print(predictedAges)

# write in a csv file
result = open('../results/resultRidge_range32_CVBarthe.csv','w')
result.write("ID,Prediction,alpha:,"+str(alpha)+"\n")
for id in range(TEST):
    result.write(str(id+1)+","+str(predictedAges[id])+"\n")
result.close()

