import csv as csv 
import numpy as np
import Activation, logReg, optim, loadData

#################################################################
# reading from csv
csv_train = csv.reader(open('../data/train.csv', 'rb'))
header = csv_train.next()
data = [[map(int, row[1:]), [int(row[0])]] for row in csv_train]

train = loadData.Data()
train.loadList(data, numClasses = 10)
train.NormalizeScale(factor = 255.0)

#################################################################
# PCA of training set
import npPCA
Z, U_reduced = npPCA.PCA(train.X, varRetained = 0.95, show = False)
train.X, train.n = Z, Z.shape[0]

#################################################################
# load testing data
csv_test = csv.reader(open('../data/test.csv', 'rb'))
header = csv_test.next()
data = [[map(int, row[0:]), [0]] for row in csv_test]

test = loadData.Data()
test.loadList(data, numClasses = 10)
test.NormalizeScale(255.0)
#test.addBiasRow()
test.X = np.transpose(U_reduced) * test.X
test.n = train.n
train.addBiasRow()
test.addBiasRow()
np.savetxt("../sub/trainX.csv", train.X, delimiter=",", fmt = '%.4f')
np.savetxt("../sub/testX.csv", test.X, delimiter = ",", fmt = '%.4f')
np.savetxt("../sub/trainy.csv", train.y, delimiter = ",", fmt = '%d')
#################################################################
######## TRAINING USING LOGISTIC REGRESSION #####################
print train.m, train.n, train.K, train.X.shape, train.y.shape
print test.m, test.n, test.K, test.X.shape
print 'Training Started'
act = Activation.sigmoid().h
model, J = logReg.trainOneVsAllGD(train, act, epochs = 500, lr = 0.8, Lambda = 0.001)
print 'Training Ended'
np.savetxt("../sub/model.csv", model, delimiter = ",", fmt = '%.4f')
import matplotlib.pyplot as plt
plt.plot(np.transpose(J))
plt.xlabel('epochs')
plt.ylabel('Cost')
np.savetxt("../sub/cost.csv", np.transpose(J), delimiter=",")
plt.legend(['0','1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.show()
test_y = logReg.predictMultiple(model, test.X, act)
np.savetxt("../sub/logReg.csv", test_y, delimiter=",", fmt = '%d')
print 'OutPut Written to File'
