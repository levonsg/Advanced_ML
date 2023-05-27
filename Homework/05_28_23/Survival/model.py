from sklearn.ensemble import GradientBoostingClassifier

class Model:

    def __init__(self):
        self.train_model = GradientBoostingClassifier(n_estimators=8000, learning_rate=0.0001, max_depth=3, random_state=42)

    def fit(self, X, y):
        self.train_model.fit(X, y)

    def predict(self, X):
        pred = self.train_model.predict(X)
        return pred

    def predict_proba(self, X):
        probs = self.train_model.predict_proba(X)
        return probs