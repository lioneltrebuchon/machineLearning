############################################################################

# Python script to make the ridge regression and output the prediction.csv

############################################################################

N_TEST = 138
ALPHA = 1

import numpy as np
from sklearn import linear_model

# Regression function (compute the prediction only if the boolean prediction==true)
def ridgeRegression(alphas, features, age, prediction=False, toPredict=np.empty(1, dtype=int)):
    # We set up the model
    modelRidge = linear_model.RidgeCV(alphas, normalize=True, cv=None)
    # We compute the model
    result = modelRidge.fit(features, age)
    # We compute the score
    scoreFinal = modelRidge.score(features, age)
    print("Coefficient: {0} Alpha: {1} Score: {2} Intercept: {3}".format(modelRidge.coef_, modelRidge.alpha_, scoreFinal, modelRidge.intercept_))

    # Prediction
    if prediction==True:
        predictionOutput = modelRidge.predict(toPredict).astype(int)
        return {'Coefficient': modelRidge.coef_, 'Alpha': modelRidge.alpha_, 'Score': scoreFinal, 'Intercept': modelRidge.intercept_, 'PredictedAges': predictionOutput}
    else:
        return {'Coefficient': modelRidge.coef_, 'Alpha': modelRidge.alpha_, 'Score': scoreFinal, 'Intercept': modelRidge.intercept_}

# Read features and ages of the train set
sectionFeatures = np.genfromtxt('features/train_section_features.csv', delimiter=",")
p2x = np.genfromtxt('features/train_p2_x.csv', delimiter=",")
p2y = np.genfromtxt('features/train_p2_y.csv', delimiter=",")
p3x = np.genfromtxt('features/train_p3_x.csv', delimiter=",")
p3y = np.genfromtxt('features/train_p3_y.csv', delimiter=",")

features = np.concatenate([sectionFeatures,np.reshape(p2x,[-1,1]),np.reshape(p2y,[-1,1]),np.reshape(p3x,[-1,1]),np.reshape(p3y,[-1,1])],1)

age = np.genfromtxt('targets.csv', delimiter="\n")

# Read features of the test set to predict
sectionFeatures = np.genfromtxt('features/test_section_features.csv', delimiter=",")
p2x = np.genfromtxt('features/test_p2_x.csv', delimiter=",")
p2y = np.genfromtxt('features/test_p2_y.csv', delimiter=",")
p3x = np.genfromtxt('features/test_p3_x.csv', delimiter=",")
p3y = np.genfromtxt('features/test_p3_y.csv', delimiter=",")

toPredictFeatures = np.concatenate([sectionFeatures,np.reshape(p2x,[-1,1]),np.reshape(p2y,[-1,1]),np.reshape(p3x,[-1,1]),np.reshape(p3y,[-1,1])],1)

# Do the regression for several alphas
alphas = [ALPHA]
print("Start Ridge regression with different alphas") 
predictedAges = ridgeRegression(alphas, features, age,True,toPredict=toPredictFeatures)['PredictedAges']

# Write in a csv file
result = open('prediction.csv','w')
result.write("ID,Prediction"+"\n")
for id in range(N_TEST):
    result.write(str(id+1)+","+str(predictedAges[id])+"\n")
result.close()

