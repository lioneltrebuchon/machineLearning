########################################################################

# Python script to output result from ridge regression

########################################################################

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import linear_model
from sklearn.linear_model import Lasso

# Prepare the different features for the regression and then create the input array features
feature1 = np.empty(278, dtype=float)
feature1 = np.genfromtxt('../results/p2_x.csv', delimiter="\n")

feature2 = np.empty(278, dtype=float)
feature2 = np.genfromtxt('../results/p3_x.csv', delimiter="\n")

feature3 = np.empty(278, dtype=float)
feature3 = np.genfromtxt('../results/p2_y.csv', delimiter="\n")

feature4 = np.empty(278, dtype=float)
feature4 = np.genfromtxt('../results/p3_y.csv', delimiter="\n")

features = np.transpose(np.array([feature1, feature2, feature3, feature4]))
age = np.empty(278, dtype=int)
age = np.genfromtxt('../data/targets.csv', delimiter="\n")

topredictfeature1 = np.empty(138, dtype=float)
topredictfeature1 = np.genfromtxt('../results/test_p2_x.csv', delimiter="\n")

topredictfeature2 = np.empty(138, dtype=float)
topredictfeature2 = np.genfromtxt('../results/test_p3_x.csv', delimiter="\n")

topredictfeature3 = np.empty(138, dtype=float)
topredictfeature3 = np.genfromtxt('../results/test_p2_y.csv', delimiter="\n")

topredictfeature4 = np.empty(138, dtype=float)
topredictfeature4 = np.genfromtxt('../results/test_p3_y.csv', delimiter="\n")

topredictfeatures = np.transpose(np.array([topredictfeature1, topredictfeature1, topredictfeature1, topredictfeature1]))

def linearregression(features, age, prediction=False, topredict=np.empty(1, dtype=int)):
    # Compute the linear regression with parameters:
    # features
    # age
    # prediction = False if we want to predict ages (by default = False)
    # topredict (features used for the prediction)

    # More info at :
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

    # We set up the model
    modelLinear = linear_model.LinearRegression(normalize=True)

    # We compute the model
    result = modelLinear.fit(features, age)

    # We compute the score
    scorefinal = modelLinear.score(features, age)

    print("Coefficient: {0} Score: {1} Intercept: {2} Residue: {3}".format(modelLinear.coef_, scorefinal, modelLinear.intercept_, modelLinear.residues_, ))

    # Prediction
    if prediction==True:
        predictionoutput = modelLinear.predict(topredict).astype(int)
        print("Prediction: {0}".format(predictionoutput))
        return {'Coefficient': modelLinear.coef_, 'Score': scorefinal, 'Intercept': modelLinear.intercept_, 'Residue': modelLinear.residues_, 'Predicted ages': predictionoutput}
    else:
        return {'Coefficient': modelLinear.coef_, 'Score': scorefinal, 'Intercept': modelLinear.intercept_, 'Residue': modelLinear.residues_}

def ridgeregression(alpha, features, age, prediction=False, topredict=np.empty(1, dtype=int)):
    # Compute the ridge regression with parameters:
    # alpha (penalizing factor, unique)
    # features
    # age
    # prediction = False if we want to predict ages (by default = False)
    # topredict (features used for the prediction)

    # More info at :
    # http://scikit-learn.org/stable/tutorial/basic/tutorial.html

    # We set up the model
    # cv : cross-validation generator (with none, Generalized Cross-Validation (efficient Leave-One-Out)
    modelRidge = linear_model.RidgeCV(alpha, normalize=True, cv=None)

    # We compute the model
    result = modelRidge.fit(features, age)

    # We compute the score
    scorefinal = modelRidge.score(features, age)

    print("Coefficient: {0} Alpha: {1} Score: {2} Intercept: {3}".format(modelRidge.coef_, alpha, scorefinal, modelRidge.intercept_))

    # Prediction
    if prediction==True:
        predictionoutput = modelRidge.predict(topredict).astype(int)
        print("Prediction: {0}".format(predictionoutput))
        return {'Coefficient': modelRidge.coef_, 'Alpha': alpha, 'Score': scorefinal, 'Intercept': modelRidge.intercept_, 'Predicted ages': predictionoutput}
    else:
        return {'Coefficient': modelRidge.coef_, 'Alpha': alpha, 'Score': scorefinal, 'Intercept': modelRidge.intercept_}

def lassoregression(alpha, features, age, prediction=False, topredict=np.empty(1, dtype=int)):
    # Compute the Lasso regression with parameters:
    # alpha (penalizing factor, unique)
    # features
    # age
    # prediction = False if we want to predict ages (by default = False)
    # topredict (features used for the prediction)

    # More info at :
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html

    # We set up the model
    # cv : cross-validation generator (with none, Generalized Cross-Validation (efficient Leave-One-Out)
    modelLasso = linear_model.LassoCV(alpha, cv=None, normalize=True)

    # We compute the model
    result = modelLasso.fit(features, age)

    # We compute the score
    scorefinal = modelLasso.score(features, age)

    print("Coefficient: {0} Alpha: {1} Score: {2} Intercept: {3}".format(modelLasso.coef_ , alpha, scorefinal, modelLasso.intercept_))

    # Prediction
    if prediction==True:
        predictionoutput = modelLasso.predict(topredict).astype(int)
        print("Prediction: {0}".format(predictionoutput))
        return {'Coefficient': modelLasso.coef_, 'Alpha': alpha, 'Score': scorefinal, 'Intercept': modelLasso.intercept_, 'Predicted ages': predictionoutput}
    else:
        return {'Coefficient': modelLasso.coef_, 'Alpha': alpha, 'Score': scorefinal, 'Intercept': modelLasso.intercept_}

# linearregression(features, age)
ridgeregression([0.1], features, age, True, topredictfeatures)
# ridgeregression([0.5], features, age)
# ridgeregression([1], features, age)
# lassoregression([0.1], features, age)
# lassoregression([0.5], features, age)
# lassoregression([1], features, age)