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

import os, sys

""" Relative path """
#train1 = nib.load("../data/set_train/train_1.nii")

""" Path from usb key """
#train1 = nib.load("/run/media/lionelt/04F6-B693/ML/data/set_train/train_1.nii")

#data = train1.get_data()

# Prepare the different features for the regression and then create the input array features
feature1 = np.empty(278, dtype=float)
feature1 = np.genfromtxt('..results/p2_x.csv', delimiter="\n")

feature2 = np.empty(278, dtype=float)
feature2 = np.genfromtxt('..results/p3_x.csv', delimiter="\n")

feature3 = np.empty(278, dtype=float)
feature3 = np.genfromtxt('..results/p2_y.csv', delimiter="\n")

feature4 = np.empty(278, dtype=float)
feature4 = np.genfromtxt('..results/p3_y.csv', delimiter="\n")

features = np.concatenate((feature1, feature2, feature3, feature4), axis=0)

age = np.empty(278, dtype=int)
age = np.genfromtxt('..results/targets.csv', delimiter="\n")

def linearregression(features, age, prediction=False, topredict=np.empty(1, dtype=int)):
    # Compute the linear regression with parameters:
    # features
    # age
    # prediction = False if we want to predict ages (by default = False)
    # topredict (features used for the prediction)

    # More info at :
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

    # We set up the model
    modelLinear = linear_model.LinearRegression()

    # We compute the model
    result = modelLinear.fit(features, age)

    # Prediction
    if prediction==True:
        predictionoutput = modelLinear.predict(topredict).astype(int)
        return {'results': result, 'predicted ages': predictionoutput}
    else:
        return result

def ridgeregression(alphas, features, age, prediction=False, topredict=np.empty(1, dtype=int)):
    # Compute the ridge regression with parameters:
    # alphas (penalizing factors)
    # features
    # age
    # prediction = False if we want to predict ages (by default = False)
    # topredict (features used for the prediction)

    # More info at :
    # http://scikit-learn.org/stable/tutorial/basic/tutorial.html

    # We set up the model
    # cv : cross-validation generator (with none, Generalized Cross-Validation (efficient Leave-One-Out)
    modelRidge = linear_model.RidgeCV(alphas, cv=None)

    # We compute the model
    result = modelRidge.fit(features, age)

    """ Useless part as the function RidgeCV already computes the cross validation
    # Cross validation
    # We compute cv times the score with different splits each time (partioning)
    scorecv = sk.cross_validation.cross_val_score(modelRidge, features, age, cv)"""

    # Prediction
    if prediction==True:
        predictionoutput = modelRidge.predict(topredict).astype(int)
        return {'results': result, 'predicted ages': predictionoutput}
    else:
        return result

def lassoregression(alphas, features, age, prediction=False, topredict=np.empty(1, dtype=int)):
    # Compute the Lasso regression with parameters:
    # alphas (penalizing factors)
    # features
    # age
    # prediction = False if we want to predict ages (by default = False)
    # topredict (features used for the prediction)

    # More info at :
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html

    # We set up the model
    # cv : cross-validation generator (with none, Generalized Cross-Validation (efficient Leave-One-Out)
    modelLasso = linear_model.LassoCV(alphas, cv=None)

    # We compute the model
    result = modelLasso.fit(features, age)

    """ Useless part as the function RidgeCV already computes the cross validation
    # Cross validation
    # We compute cv times the score with different splits each time (partioning)
    scorecv = sk.cross_validation.cross_val_score(modelRidge, features, age, cv)"""

    # Prediction
    if prediction==True:
        predictionoutput = modelLasso.predict(topredict).astype(int)
        return {'results': result, 'predicted ages': predictionoutput}
    else:
        return result

linearregression(features, age)
ridgeregression([0.1, 0.5, 1], features, age)
lassoregression([0.1, 0.5, 1], features, age)
