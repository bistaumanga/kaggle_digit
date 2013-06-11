import csv as csv 
import numpy as np

knn = csv.reader(open('../data/knn_benchmark.csv', 'rb'))
knnData = np.array([row for row in knn])

rf = csv.reader(open('../data/rf_benchmark.csv', 'rb'))
rfData = np.array([row for row in rf])

op = csv.reader(open('../sub/svm2.csv', 'rb'))
data = np.array([row for row in op])

compKnn = knnData == data
compRf = rfData == data
print np.sum(compKnn)/ 28000.0, np.sum(compRf)/ 28000.0