import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris



X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24)

class tree():
    def __init__(self):
        pass

    def best_split(self, X , y):
        best_feature = None
        best_threshold = None
        best_gini = 1

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold

                left_gini = self.gini(y[left_mask])
                right_gini = self.gini(y[right_mask])
                split_gini = len(y[left_mask]) / len(y) * left_gini + len(y[right_mask]) / len(y) * right_gini

                if split_gini < best_gini:
                    best_feature = feature
                    best_threshold = threshold
                    best_gini = split_gini
        return best_feature, best_threshold
    def fit(self, X, y):
        if len(np.unique(y)) == 1:

            return np.unique(y)[0]
        best_feature, best_threshold = self.best_split(X, y)
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold
        Tr = self.fit(X[left_mask], y[left_mask])

        Fl = self.fit(X[right_mask], y[right_mask])
        return {'feature': best_feature, 'value': best_threshold, 'true': Tr, 'false': Fl, }
    def gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        pk = counts/len(y)
        return np.sum(pk * (1-pk))



T = tree()
D = T.fit(X_train, y_train)
print(D)