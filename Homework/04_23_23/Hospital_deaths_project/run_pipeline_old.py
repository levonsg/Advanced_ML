import argparse
import pandas as pd
from preprocessor import Preprocessor
from model import Model
import numpy as np
import pickle
import json
#from sklearn.base import BaseEstimator, TransformerMixin


class Pipeline:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model



    def run(self, X, test=False):

        if not test:
            y = X['In-hospital_death']
            X = X.drop('In-hospital_death', axis=1)
            self.preprocessor.fit(X)
            X = self.preprocessor.transform(X)

            self.model.fit(X, y)

            pickle.dump(self.preprocessor, open('preprocessor.pkl', 'wb'))
            pickle.dump(self.model, open('model.pkl', 'wb'))
        else:

            preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
            model = pickle.load(open('model.pkl', 'rb'))

            X = preprocessor.transform(X)

            predict_probas = model.predict_proba(X)

            threshold = 0.5
            predictions = (predict_probas[:, 1] > threshold).astype(int)
            results = {'predict_probas': predictions.tolist(), 'threshold': threshold}
            with open('predictions.json', 'w') as f:
                json.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--inference', action='store_true', help='Run in inference mode')
    args = parser.parse_args()

    data = pd.read_csv(args.data_path)

    preprocessor = Preprocessor()
    model = Model()

    pipeline = Pipeline(preprocessor, model)
    pipeline.run(data, test=args.inference)