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
from sklearn import preprocessing

# Input (features and target) of the regression
p2x = np.genfromtxt('../features/train_p2_x.csv', delimiter=",")
p2y = np.genfromtxt('../features/train_p2_y.csv', delimiter=",")
p3x = np.genfromtxt('../features/train_p3_x.csv', delimiter=",")
p3y = np.genfromtxt('../features/train_p3_y.csv', delimiter=",")
p2x = preprocessing.scale(p2x)
p2y = preprocessing.scale(p2y)
p3x = preprocessing.scale(p3x)
p3y = preprocessing.scale(p3y)

sectionFeatures = np.genfromtxt('../features/train_section_features.csv', delimiter=",")
sectionFeatures = preprocessing.scale(sectionFeatures)

features = np.concatenate([np.reshape(p2x,[-1,1]),np.reshape(p2y,[-1,1]),np.reshape(p3x,[-1,1]),np.reshape(p3y,[-1,1]),sectionFeatures],1)

target = np.genfromtxt('../data/targets.csv', delimiter=",")

# Features for the prediction
# Read features of the test set to predict
p2x = np.genfromtxt('../features/test_p2_x.csv', delimiter=",")
p2y = np.genfromtxt('../features/test_p2_y.csv', delimiter=",")
p3x = np.genfromtxt('../features/test_p3_x.csv', delimiter=",")
p3y = np.genfromtxt('../features/test_p3_y.csv', delimiter=",")
p2x = preprocessing.scale(p2x)
p2y = preprocessing.scale(p2y)
p3x = preprocessing.scale(p3x)
p3y = preprocessing.scale(p3y)

sectionFeatures = np.genfromtxt('../features/test_section_features.csv', delimiter=",")
sectionFeatures = preprocessing.scale(sectionFeatures)

toPredictFeatures = np.concatenate([np.reshape(p2x,[-1,1]),np.reshape(p2y,[-1,1]),np.reshape(p3x,[-1,1]),np.reshape(p3y,[-1,1]),sectionFeatures],1)

def ridgeRegression(alphas, features, target, prediction=False, toPredict=np.empty(1, dtype=int)):
    # More info at :
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html

    # We set up the model
    # cv : cross-validation generator (with none, Generalized Cross-Validation (efficient Leave-One-Out)
    modelRidge = linear_model.RidgeCV(alphas, normalize=True, cv=None, store_cv_values = True)

    # We compute the model
    result = modelRidge.fit(features, target)

    # We compute the score
    scoreFinal = modelRidge.score(features, target)

    # Cross Val Score
    #scoreCV = np.sum(cross_val_score(modelRidge, features, age, cv=10))/10
    scoreCV = 0

    #print("Coefficient: {0} Alpha: {1} Score: {2} Intercept: {3}".format(modelRidge.coef_, modelRidge.alpha_, scoreFinal, modelRidge.intercept_))

    # Prediction
    if prediction==True:
        predictionOutput = modelRidge.predict(toPredict)
        return {'Coefficient': modelRidge.coef_, 'Alpha': modelRidge.alpha_, 'Score': scoreFinal, 'Intercept': modelRidge.intercept_, 'Predicted': predictionOutput, 'scoreCV': scoreCV, 'valuesCV': modelRidge.cv_values_}
    else:
        return {'Coefficient': modelRidge.coef_, 'Alpha': modelRidge.alpha_, 'Score': scoreFinal, 'Intercept': modelRidge.intercept_, 'scoreCV': scoreCV, 'valuesCV': modelRidge.cv_values_}

# compute the regression for several alphas
#alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
#alphas = np.linspace(0.1, 1.7, 20)
alphas = [0.5]

#listCV = np.empty([len(alphas)], dtype=float)
#j = 0

for alpha in alphas:
    #coefficient, alpha, score, intercept, predicted
    print("Start Ridge regression with different alphas "+str(alpha))
    results = ridgeRegression([alpha], features, target, True, toPredict=toPredictFeatures)
    predicted = results['Predicted']

    #print(predicted)
    print(results['valuesCV'])
    print(results['valuesCV'].shape)

    predictedRounded = [[0 for i in xrange(3)] for i in xrange(TEST)]

    # Round up predictions
    for id in range(TEST):
        predictedRounded[id][0] = round(predicted[id][0])
        if predictedRounded[id][0]<=0:
            predictedRounded[id][0] = 'FALSE'
        else:
            predictedRounded[id][0] = 'TRUE'

        predictedRounded[id][1] = round(predicted[id][1])
        if predictedRounded[id][1]<=0:
            predictedRounded[id][1] = 'FALSE'
        else:
            predictedRounded[id][1] = 'TRUE'

        predictedRounded[id][2] = round(predicted[id][2])
        if predictedRounded[id][2]<=0:
            predictedRounded[id][2] = 'FALSE'
        else:
            predictedRounded[id][2] = 'TRUE'
    
    # write in a csv file
    result = open('../results/ridgeNoPrepro'+str(alpha)+'.csv','w')
    result.write("ID,Sample,Label,Predicted"+"\n")
    for id in range(TEST*3):
        if id%3==0:
            result.write(str(id)+","+str(id/3)+",gender,"+predictedRounded[id/3][0]+','+str(predicted[id/3][0])+"\n")
        elif id%3==1:
            result.write(str(id)+","+str(id/3)+",age,"+predictedRounded[id/3][1]+','+str(predicted[id/3][1])+"\n")
        elif id%3==2:
            result.write(str(id)+","+str(id/3)+",health,"+predictedRounded[id/3][2]+','+str(predicted[id/3][2])+"\n")
        else:
            print("Error during prediction for id: "+str(id))
            result.write("ERROR,ERROR,ERROR,ERROR"+"\n")
    result.close()
