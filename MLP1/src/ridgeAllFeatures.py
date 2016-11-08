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
sliceFeatures48 = np.genfromtxt('../results/sliceTrainFeatures48.csv', delimiter=",")
sliceFeatures32 = np.genfromtxt('../results/sliceTrainFeatures32.csv', delimiter=",")
sliceFeatures64 = np.genfromtxt('../results/sliceTrainFeatures64.csv', delimiter=",")
p2x = np.genfromtxt('../results/p2_x.csv', delimiter=",")
p2y = np.genfromtxt('../results/p2_y.csv', delimiter=",")
p3x = np.genfromtxt('../results/p3_x.csv', delimiter=",")
p3y = np.genfromtxt('../results/p3_y.csv', delimiter=",")

features = np.concatenate([sliceFeatures48, sliceFeatures32, sliceFeatures64 ,np.reshape(p2x,[-1,1]),np.reshape(p2y,[-1,1]),np.reshape(p3x,[-1,1]),np.reshape(p3y,[-1,1])],1)

#print(features)
#features = features[0:-1,0:features.shape[0]]
age = np.genfromtxt('../data/targets.csv', delimiter="\n")

# Features for the prediction
sliceFeatures48 = np.genfromtxt('../results/sliceTestFeatures48.csv', delimiter=",")
sliceFeatures32 = np.genfromtxt('../results/sliceTestFeatures32.csv', delimiter=",")
sliceFeatures64 = np.genfromtxt('../results/sliceTestFeatures64.csv', delimiter=",")
p2x = np.genfromtxt('../results/test_p2_x.csv', delimiter=",")
p2y = np.genfromtxt('../results/test_p2_y.csv', delimiter=",")
p3x = np.genfromtxt('../results/test_p3_x.csv', delimiter=",")
p3y = np.genfromtxt('../results/test_p3_y.csv', delimiter=",")

toPredictFeatures = np.concatenate([sliceFeatures48, sliceFeatures32, sliceFeatures64 ,np.reshape(p2x,[-1,1]),np.reshape(p2y,[-1,1]),np.reshape(p3x,[-1,1]),np.reshape(p3y,[-1,1])],1)

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
        predictionOutput = np.round(modelRidge.predict(toPredict))
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
alphas = [1.2]
#alphas = np.linspace(0.00000000001,0.0000001,10001)

#results = ridgeRegression(alphas, features, age)

#coefficient, alpha, score, intercept, predictedAges
print("Start Ridge regression with different alphas") 
results = ridgeRegression(alphas, features, age,True,toPredict=toPredictFeatures)
alpha=results['Alpha']
predictedAges=results['PredictedAges']

print(alpha)
print(predictedAges)


# write in a csv file
result = open('../results/resultRidge_allRangesWithPeaks_alpha'+str(alpha)+'.csv','w')
result.write("ID,Prediction"+"\n")
for id in range(TEST):
    #print(id)
    result.write(str(id+1)+","+str(predictedAges[id])+"\n")
result.close()

