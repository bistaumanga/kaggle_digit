import csv as csv 
import numpy as np
import Activation, logReg, optim, loadData

# reding from csv
csv_train = csv.reader(open('../data/train.csv', 'rb'))
header = csv_train.next()
data=[]
for row in csv_train:      
    data.append([map(int, row[1:]), [int(row[0])]])

train = loadData.Data()
train.loadList(data, numClasses = 10)
train.NormalizeScale(factor = 255.0)
#train.addBiasRow()

import npPCA
Z = npPCA.PCA(train.X, varRetained = 0.9, show = True)


#print train.X.shape

# reding from csv
# csv_test = csv.reader(open('../data/test.csv', 'rb'))
# header = csv_test.next()
# data=[]
# for row in csv_test:      
#     data.append([map(int, row[0:]), [0]])

# test = loadData.Data()
# test.loadList(data, numClasses = 10)
# test.NormalizefeatureScale()
# test.addBiasRow()
# print test.m, test.n, test.K




# # and function of 3 variables
# pat1 = [[[0, 0, 0], [0]],
# 	[[0, 0, 1], [1]],
# 	[[0, 1, 0], [1]],
# 	[[0, 1, 1], [1]],
# 	[[1, 0, 0], [2]],
# 	[[1, 0, 1], [3]],
# 	[[1, 1, 0], [3]],
# 	[[1, 1, 1], [3]],
# 	]

# d1 = Data()
# d1.loadList(pat1, numClasses = 4)
# #print d1.y
# act = sigmoid().h # our activation function is simgmoid
# model, J = trainOneVsAllGD(d1, act,epochs = 5000, lr = 0.25)
# #print d1.y
# plt.plot(np.transpose(J))
# plt.show()

# print predictMultiple(model, d1.X, act)