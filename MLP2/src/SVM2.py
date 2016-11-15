########################################################################

# Python script to output result from SVM classification

########################################################################

TEST=138
TRAIN=278

import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import svm

# Input (features and targets) of the regression
feature1 = np.genfromtxt('../features/train_p2_x.csv', delimiter="\n")
feature2 = np.genfromtxt('../features/train_p3_x.csv', delimiter="\n")
feature3 = np.genfromtxt('../features/train_p2_y.csv', delimiter="\n")
feature4 = np.genfromtxt('../features/train_p3_y.csv', delimiter="\n")
features = np.transpose(np.array([feature1, feature2, feature3, feature4]))
targets = np.genfromtxt('../data/targets.csv', delimiter="\n").astype(int)


# Features for the prediction
toPredictFeatures1 = np.genfromtxt('../features/test_p2_x.csv', delimiter="\n")
toPredictFeatures2 = np.genfromtxt('../features/test_p3_x.csv', delimiter="\n")
toPredictFeatures3 = np.genfromtxt('../features/test_p2_y.csv', delimiter="\n")
toPredictFeatures4 = np.genfromtxt('../features/test_p3_y.csv', delimiter="\n")
toPredictFeatures = np.transpose(np.array([toPredictFeatures1, toPredictFeatures2, toPredictFeatures3, toPredictFeatures4]))

def svmclassification(features, targets, C=1, kernel='rbf', gamma='auto', decision_function_shape=None, prediction=False, toPredict=np.empty(1, dtype=int)):
    # More info at:
    # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

    # Parameters:
    # Constant C : penalty parameter of the error term
    # kernel function used for the classification. It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable
    # gamma: kernel coefficient (float) for 'rbf', 'poly' and 'sigmoid'.If gamma is 'auto' then 1 / n_features will be used instead.
    # decision_function_shape returns a one-vs-rest (ovr) or the one-vs-one (ovo) decision 		# function (by default with None)

    # We set up the model
    modelSVM = svm.SVC(C=C, kernel=kernel, decision_function_shape=decision_function_shape)

    # We compute the model
    modelSVM.fit(features, targets)

    # Compute the distance of the samples X to the separating hyperplane.
    # Return : array-like, shape (n_samples,)
    dist = modelSVM.decision_function(features)

    # We compute the score (mean accuracy wrt. to the real output targets)
    # Not very relevant as we compute a score over the training set.
    scoreFinal = modelSVM.score(features, targets)

    # Attributs:
    # Support vectors support_vectors_ : array-like, shape = [n_SV, n_features]
    # with indices support_ : array-like, shape = [n_SV]
    # Number of support vectors for each class n_support_ : array-like, dtype=int32, shape = [2]
    # Coefficients of the support vector in the decision function dual_coef_ : array, shape = [1, n_SV]
    # Constants in decision function intercept_ : array, shape = [1]

    print("SV: {0} SV indices: {1} SV repartition: {2} SV coefficients: {3} Intercept: {4} Score: {5}".format(modelSVM.support_vectors_, modelSVM.support_, modelSVM.n_support_, modelSVM.dual_coef_, modelSVM.intercept_, scoreFinal))

    # Prediction
    if prediction==True:
        predictionOutput = (modelSVM.predict(toPredict))
        return {'SV': modelSVM.support_vectors_, 'SV indices': modelSVM.support_, 'SV repartition': modelSVM.n_support_, 'SV coefficients': modelSVM.dual_coef_, 'Intercept': modelSVM.intercept_, 'Score': scoreFinal, 'PredictedClass': predictionOutput}
    else:
        return {'SV': modelSVM.support_vectors_, 'SV indices': modelSVM.support_, 'SV repartition': modelSVM.n_support_, 'SV coefficients': modelSVM.dual_coef_, 'Intercept': modelSVM.intercept_, 'Score': scoreFinal}

# compute the regression for several C
#c = np.linspace(0.00000000001,0.0000001,10001)
c = 1.0

print("Start SVM classification with C = "+str(c))
prediction = False
svmclassification(features, targets, c, kernel='rbf', gamma='auto', decision_function_shape=None, prediction=False, toPredict=np.empty(1, dtype=int))

# write in a csv file
if prediction==True:
    PredictedClass = results['PredictedClass']
    result = open('../results/prediction.csv','w')
    result.write("ID,Prediction,C:,"+str(C)+"\n")
    for id in range(TEST):
        result.write(str(id+1)+","+str(PredictedClass[id])+"\n")
    result.close()

