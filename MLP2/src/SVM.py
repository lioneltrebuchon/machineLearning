from sklearn import svm

sklearn.svm.SVC(C=1.0, kernel='rbf', decision_function_shape=None)
	# More info at:
	# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

	# Parameters:
	# Constant C which ?
	# kernel function used for the classification. It must be one of ‘linear’, ‘poly’, ‘rbf’, 		# ‘sigmoid’, ‘precomputed’ or a callable
	# decision_function_shape returns a one-vs-rest (ovr) or the one-vs-one (ovo) decision 		# function (by default with None)

	# Attributs:
	# Support vectors support_vectors_ : array-like, shape = [n_SV, n_features]
	# with indices support_ : array-like, shape = [n_SV]
	# Number of support vectors for each class n_support_ : array-like, dtype=int32, shape = [2]
	# Coefficients of the support vector in the decision function dual_coef_ : array, shape = [1, n_SV]
	# Constants in decision function intercept_ : array, shape = [1]

# Training samples
# X : array-like, shape (n_samples, n_features)

# Target variable of X
# Y : array-like, shape (n_samples)

# Test samples
# Z : array-like, shape = (n_samples, n_features)

# Target variable of Z
# W : array-like, shape = (n_samples)

# Compute the distance of the samples X to the separating hyperplane.
# Return X : array-like, shape (n_samples, 1)
decision_function(X)

# Fit the SVM model according to the given training data X and the target variable Y
fit(X, Y)

# Perform classification on samples Z.
y_pred = predict(Z)
# Return class labels (return +1 or -1 for one-class model).
# y_pred : array, shape (n_samples)

# Returns the mean accuracy wrt. W on the given test data and labels.
score = score(Z, W)
# score : float
