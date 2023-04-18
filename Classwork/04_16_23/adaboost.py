import numpy as np
from copy import deepcopy

class AdaBoost:
    def __init__(self, classifier, m):
        self.classifier = classifier
        self.m = m
        self.weights = None
        self.alphas = np.zeros(self.m)
        self.classifiers = []

    def fit(self, X, y):
        self.weights = np.ones(X.shape[0]) / X.shape[0]

        for m in range(self.m):
            current_classifier = deepcopy(self.classifier)
            current_classifier.fit(X, y, sample_weight=self.weights)
            current_pred = current_classifier.predict(X)

            current_error = np.sum(self.weights[current_pred != y]) / np.sum(self.weights)

            current_alpha = np.log((1 - current_error) / current_error)
            self.alphas[m] = current_alpha

            self.weights = self.weights * np.exp(current_alpha * (current_pred!=y))

            self.classifiers.append(current_classifier)

    def predict(self, X):
        prediction = []
        for x in X:
            current_predict = 0

            for i, classifier in enumerate(self.classifiers):
                current_predict += self.alphas[i] * classifier.predict(x.reshape(1, -1))

            prediction.append(np.sign(current_predict)[0])

        return np.array(prediction)


    def score(self, X, y):
        return (y == self.predict(X)).mean()

import pandas as pd
data = pd.read_csv('../../Desktop/ACA/week 4/classification.csv')
X = data.drop(['default', 'ed'], axis=1).to_numpy()
# X = X.select_dtyp
y = data['default'].to_numpy()
y[y==0] = -1

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=2)
clf = AdaBoost(tree, 10)
clf.fit(X, y)
# for i in range(10):
#     print(clf.classifiers[i].score(X, y))

print(clf.score(X,y))

# from sklearn.ensemble import AdaBoostClassifier
#
# model = AdaBoostClassifier(tree, n_estimators = 10)
# model.fit(X, y)
# print(model.predict(X))

