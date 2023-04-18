import argparse
import pandas as pd
from preprocessor import Preprocessor
from model import Model
import numpy as np
import pickle
import json

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

            # save preprocessor and model in training mode
            pickle.dump(self.preprocessor, open('preprocessor_train.pkl', 'wb'))
            # self.model.save('model_train.pkl')
            pickle.dump(self.model, open('model_train.pkl', 'wb'))
        else:

            loaded_preprocessor = pickle.load(open('preprocessor_train.pkl', 'rb'))

            X = loaded_preprocessor.transform(X.values)

            loaded_model = pickle.load(open('model_train.pkl', 'rb'))
            predict_probas = loaded_model.predict_proba(X)
            threshold = 0.5  # TODO: adjust threshold based on validation set
            predictions = (predict_probas[:, 1] > threshold).astype(int)
            results = {'predict_probas': predict_probas.tolist(), 'threshold': threshold}
            with open('predictions.json', 'w') as f:
                json.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--inference', action='store_true', help='Run in inference mode')
    args = parser.parse_args()

    # load data
    data = pd.read_csv(args.data_path)

    # initialize preprocessor and model
    preprocessor = Preprocessor()
    model = Model()

    # run pipeline
    pipeline = Pipeline(preprocessor, model)
    pipeline.run(data, test=args.inference)