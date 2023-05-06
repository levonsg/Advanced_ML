import pickle
from sklearn.linear_model import LogisticRegression

class Model:

    def __init__(self, inference=0):
        # self.inference=inference
        # with open('Predict_deaths.sav', 'rb') as f:
        #     self.predict_deaths = pickle.load(f)
        self.train_model = LogisticRegression(random_state=0)

    def fit(self, X, y):
        self.train_model.fit(X, y)

    def predict(self, X):
        # if self.inference==0:
        #     ada_pred = self.train_model.predict(X)
        # else:
        #     ada_pred = self.predict_deaths.predict(X)
        # return ada_pred
        ada_pred = self.train_model.predict(X)
        return ada_pred

    def predict_proba(self, X):
        # if self.inference==0:
        #     ada_probs = self.train_model.predict_proba(X)
        #     return ada_probs
        # else:
        #     ada_probs = self.predict_deaths.predict_proba(X)
        #     return ada_probs
        ada_probs = self.train_model.predict_proba(X)
        return ada_probs