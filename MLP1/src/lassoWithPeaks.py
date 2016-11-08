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

# feature1 = np.genfromtxt('../results/p2_x.csv', delimiter="\n")
# feature2 = np.genfromtxt('../results/p3_x.csv', delimiter="\n")
# feature3 = np.genfromtxt('../results/p2_y.csv', delimiter="\n")
# feature4 = np.genfromtxt('../results/p3_y.csv', delimiter="\n")
# features5 = np.transpose(np.array([feature1, feature2, feature3, feature4]))

# Input (features and age) of the regression
sliceFeatures = np.genfromtxt('../results/sliceTrainFeatures48.csv', delimiter=",")
p2x = np.genfromtxt('../results/p2_x.csv', delimiter=",")
p2y = np.genfromtxt('../results/p2_y.csv', delimiter=",")
p3x = np.genfromtxt('../results/p3_x.csv', delimiter=",")
p3y = np.genfromtxt('../results/p3_y.csv', delimiter=",")

features = np.concatenate([sliceFeatures,np.reshape(p2x,[-1,1]),np.reshape(p2y,[-1,1]),np.reshape(p3x,[-1,1]),np.reshape(p3y,[-1,1])],1)

#print(features)
#features = features[0:-1,0:features.shape[0]]
age = np.genfromtxt('../data/targets.csv', delimiter="\n")

# Features for the prediction
sliceFeatures = np.genfromtxt('../results/sliceTestFeatures48.csv', delimiter=",")
p2x = np.genfromtxt('../results/test_p2_x.csv', delimiter=",")
p2y = np.genfromtxt('../results/test_p2_y.csv', delimiter=",")
p3x = np.genfromtxt('../results/test_p3_x.csv', delimiter=",")
p3y = np.genfromtxt('../results/test_p3_y.csv', delimiter=",")

toPredictFeatures = np.concatenate([sliceFeatures,np.reshape(p2x,[-1,1]),np.reshape(p2y,[-1,1]),np.reshape(p3x,[-1,1]),np.reshape(p3y,[-1,1])],1)

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
        predictionOutput = np.round(modelLasso.predict(toPredict))
        return {'Coefficient': modelLasso.coef_, 'Alpha': modelLasso.alpha_, 'Score': scoreFinal, 'Intercept': modelLasso.intercept_, 'PredictedAges': predictionOutput}
    else:
        return {'Coefficient': modelLasso.coef_, 'Alpha': modelLasso.alpha_, 'Score': scoreFinal, 'Intercept': modelLasso.intercept_}

# compute the regression for several alphas
alphas = [0.1]
#alphas = [0.001,0.01,0.1,0.5,1,5,10,50,100]
#alphas = np.linspace(0.00000000001,0.0000001,10001)

#coefficient, alpha, score, intercept, predictedAges
print("Start Lasso regression with different alphas"+str(alphas))
results = lassoRegression(alphas, features, age, True, toPredict=toPredictFeatures)
alpha = results['Alpha']
predictedAges = results['PredictedAges']

# write in a csv file
result = open('../results/resultLasso_range48withpeaks_alpha'+str(alpha)+'.csv','w')
result.write("ID,Prediction\n")
for id in range(TEST):
    result.write(str(id+1)+","+str(predictedAges[id])+"\n")
result.close()
