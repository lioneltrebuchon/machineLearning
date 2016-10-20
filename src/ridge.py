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

topredictfeature1 = np.empty(1, dtype=float)
topredictfeature1 = np.genfromtxt('', delimiter="\n")

topredictfeature2 = np.empty(1, dtype=float)
topredictfeature2 = np.genfromtxt('', delimiter="\n")

topredictfeature3 = np.empty(1, dtype=float)
topredictfeature3 = np.genfromtxt('', delimiter="\n")

topredictfeature4 = np.empty(1, dtype=float)
topredictfeature4 = np.genfromtxt('', delimiter="\n")

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
        return {'results': result, 'predicted ages': predictionoutput}
        print("Prediction: {0}".format(predictionoutput.C))
    else:
        return result

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
        return {'results': result, 'predicted ages': predictionoutput}
        print("Prediction: {0}".format(predictionoutput.C))
    else:
        return result

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
        return {'results': result, 'predicted ages': predictionoutput}
        print("Prediction: {0}".format(predictionoutput.C))
    else:
        return result

linearregression(features, age)
ridgeregression([0.1], features, age)
ridgeregression([0.5], features, age)
ridgeregression([1], features, age)
lassoregression([0.1], features, age)
lassoregression([0.5], features, age)
lassoregression([1], features, age)

print(modelLasso.coef_)

# peaksFile = open('../results/regressionresults.csv','w')
# peaksFile.write("Coefficient:")
# peaksFile.write("\n")
# peaksFile.close()
#
# import xlwt
# book = xlwt.Workbook(encoding="utf-8")
# sheet1 = book.add_sheet("Sheet 1")
#
# sheet1.write(2, 1, "Coefficient 1")
# sheet1.write(3, 1, "Coefficient 2")
# sheet1.write(4, 1, "Coefficient 3")
# sheet1.write(5, 1, "Coefficient 4")
# sheet1.write(6, 1, "Alpha")
# sheet1.write(7, 1, "Score")
# sheet1.write(8, 1, "Intercept")
# sheet1.write(9, 1, "Residue")
#
# sheet1.write(1, 2, "Linear")
# sheet1.write(1, 3, "Ridge")
# sheet1.write(1, 4, "Ridge")
# sheet1.write(1, 5, "Ridge")
# sheet1.write(1, 6, "Lasso")
# sheet1.write(1, 7, "Lasso")
# sheet1.write(1, 8, "Lasso")
#
# book.save("trial.xls")