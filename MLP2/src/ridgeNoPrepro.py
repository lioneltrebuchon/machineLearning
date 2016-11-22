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

# Input (features and age) of the regression
features = np.genfromtxt('../features/train_section_features.csv', delimiter=",")
age = np.genfromtxt('../data/targets.csv', delimiter="\n")

# Features for the prediction
toPredictFeatures = np.genfromtxt('../features/test_section_features.csv', delimiter=",")

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
        predictionOutput = modelRidge.predict(toPredict)
        return {'Coefficient': modelRidge.coef_, 'Alpha': modelRidge.alpha_, 'Score': scoreFinal, 'Intercept': modelRidge.intercept_, 'PredictedAges': predictionOutput}
    else:
        return {'Coefficient': modelRidge.coef_, 'Alpha': modelRidge.alpha_, 'Score': scoreFinal, 'Intercept': modelRidge.intercept_}

# compute the regression for several alphas
alphas = [5]
#alphas = np.linspace(1, 10000, 10000)

#coefficient, alpha, score, intercept, predictedAges
print("Start Ridge regression with different alphas"+str(alphas))
results = ridgeRegression(alphas, features, age, True, toPredict=toPredictFeatures)
alpha = results['Alpha']
predictedAges = results['PredictedAges']

print(predictedAges)

# write in a csv file
result = open('../results/ridgeNoPrepro'+str(alpha)+'.csv','w')
result.write("ID,Prediction"+"\n")
for id in range(TEST):
	if predictedAges[id]>1:
		result.write(str(id+1)+","+str(1)+"\n")
	elif predictedAges[id]<0:
		result.write(str(id+1)+","+str(0)+"\n")
	else:
        	result.write(str(id+1)+","+str(predictedAges[id])+"\n")
result.close()
