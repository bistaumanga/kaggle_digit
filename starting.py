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
train.normalize()
train.addBiasRow()
print train.m, train.n, train.K

# reding from csv
csv_test = csv.reader(open('../data/test.csv', 'rb'))
header = csv_test.next()
data=[]
for row in csv_test:      
    data.append([map(int, row[0:]), [0]])

test = loadData.Data()
test.loadList(data, numClasses = 10)
test.normalize()
test.addBiasRow()
print test.m, test.n, test.K
