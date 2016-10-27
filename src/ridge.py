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

# Input (features and age) of the regression
features = np.genfromtxt('../results/sliceTrainFeatures32.csv', delimiter=",")
#print(features)
#features = features[0:-1,0:features.shape[0]]
age = np.genfromtxt('../data/targets.csv', delimiter="\n")


# Features for the prediction
toPredictFeatures = np.genfromtxt('../results/sliceTestFeatures32.csv', delimiter=",")

def ridgeRegression(alphas, features, age, prediction=False, toPredict=np.empty(1, dtype=int)):
    # More info at :
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html
    print("Test")
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
'''
print(features)
print(toPredictFeatures)
print(features.shape)
print(toPredictFeatures.shape)
'''
# compute the regression for several alphas
#alphas = [0.001,0.01,0.1,0.5,1,5,10,50,100]
alphas = np.linspace(0.00000000001,0.0000001,10001)

#results = ridgeRegression(alphas, features, age)

#coefficient, alpha, score, intercept, predictedAges
print("Start Ridge regression with different alphas") 

results = ridgeRegression(alphas, features, age,True,toPredict=toPredictFeatures)
alpha=results['Alpha']
predictedAges=results['PredictedAges']

print(alpha)
print(predictedAges)


# write in a csv file
result = open('../results/resultRidge3_range32.csv','w')
result.write("ID,Prediction,alpha:,"+str(alpha)+"\n")
for id in range(TEST):
    #print(id)
    result.write(str(id+1)+","+str(predictedAges[id])+"\n")
result.close()

