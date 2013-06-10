import csv as csv 
import numpy as np
import Activation, logReg, optim, loadData

#################################################################
# reading from csv
print 'Loading Training Data'
csv_train = csv.reader(open('../../data/train.csv', 'rb'))
header = csv_train.next()
data = [[map(int, row[1:]), [int(row[0])]] for row in csv_train]

train = loadData.Data()
train.loadList(data, numClasses = 10)
train.NormalizeScale(factor = 255.0)

#################################################################
# PCA of training set
print 'Performing PCA - Principal COmponent Analysis'
import npPCA
Z, U_reduced = npPCA.PCA(train.X, varRetained = 0.95, show = False)
train.X, train.n = Z, Z.shape[0]

#################################################################
# load testing data
print 'Loading Test Data'
csv_test = csv.reader(open('../../data/test.csv', 'rb'))
header = csv_test.next()
data = [[map(int, row[0:]), [0]] for row in csv_test]

test = loadData.Data()
test.loadList(data, numClasses = 10)
test.NormalizeScale(255.0)

#################################################################
test.X = np.transpose(U_reduced) * test.X
test.n = train.n
train.addBiasRow()
test.addBiasRow()

#################################################################
######## TRAINING USING LOGISTIC REGRESSION #####################

print 'Training Started'
act = Activation.sigmoid().h
model, J = logReg.trainOneVsAllGD(train, act, epochs = 200, lr = 0.8, Lambda = 0.003)
print 'Training Ended'

import matplotlib.pyplot as plt
plt.plot(np.transpose(J))
plt.xlabel('epochs')
plt.ylabel('Cost')
plt.title('Cost in each epochs for each Logistic Regression model')
plt.legend(['0','1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.show()

print 'Prediction for Test Data'
test_y = logReg.predictMultiple(model, test.X, act)
print 'Writing predictions to file'
np.savetxt("../sub/logReg.csv", test_y, delimiter=",", fmt = '%d')
print 'OutPut Written to File'