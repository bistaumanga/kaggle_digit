import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA

print 'loading train data'
train = pd.read_csv('../../data/train.csv')
train = train.astype('float64')

pca_xfrms = []
SVMs = []

# compute indexes where to split the data
ixs = np.arange(train.shape[0])
splits = np.split(ixs, [14000, 28000])

params = [{'gamma': [0.1, 1e-2, 1e-3], 'C': [10, 100, 1000]}]

print 'training started'
# use all data for training
for s in splits:
    pca = PCA(n_components = 101)
    # get training subset
    X = train.ix[s,1:].copy()
    y = train.ix[s,0].copy()

    X = pca.fit_transform(X / 255.0)
    # train the classifier
    svm = GridSearchCV( SVC(), params, verbose=1 ).fit(X, y)

    pca_xfrms.append(pca)
    SVMs.append(svm)
print 'training ended'

print 'loading test data'

test = pd.read_csv('../../data/test.csv')
test = test.astype('float64')

preds = np.zeros((3, test.shape[0]))

i = 0
print 'predicting'
for  pca, svm in zip(pca_xfrms, SVMs):
    Xt = test.copy()
    preds[i] = svm.predict(pca.transform(Xt/ 255.0))
    i += 1
    
p = [np.bincount(x).argmax() for x in preds.T.astype(int)]

print 'writing to file'
with open('../../sub/svm2.csv', 'w') as of:
    of.write('\n'.join(str(x) for x in p))
print 'predictions written'