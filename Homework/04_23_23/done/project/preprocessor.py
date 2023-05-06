import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler


class Preprocessor:
    def __init__(self,missing_threshold = 0.9):
        self.missing_threshold = missing_threshold
        
        self.columns_to_drop_ = None
        self.columns_with_NaN = None
        self.imputer_ = None
        self.scaler = MinMaxScaler()
        self.oversampler = RandomOverSampler()

    def fit(self,X):
        self.columns_to_drop_ = X.columns[X.isnull().mean() > self.missing_threshold].to_list()
        X.drop(columns = self.columns_to_drop_, axis = 1, inplace = True)

        self.imputer_ = KNNImputer(n_neighbors=5, weights='distance')
        self.imputer_.fit(X)    
        self.scaler.fit(X)

        return self

    
    def transform(self,X, y = None, oversampler = False):
        
        if all(col in X.columns for col in self.columns_to_drop_):
            X.drop(columns=self.columns_to_drop_, axis = 1, inplace = True)

        X = pd.DataFrame(self.imputer_.transform(X), columns = X.columns)
        X_scaled = self.scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns = X.columns)

        if oversampler:
            X, y = self.oversampler.fit_resample(X,y)
        
        return X, y
    
    def fit_transform(self,X,y = None):
        self.fit(X)
        return self.transform(X,y,oversampler=True)


if __name__ == '__main__':
    df = pd.read_csv('hospital_deaths_train.csv')
    target_variable = 'In-hospital_death'
    
    X = df.drop(['recordid',target_variable], axis = 1)
    y = df[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=42, stratify = y)

    preprocessor = Preprocessor()

    X_train_filled, y_train_filled = preprocessor.fit_transform(X_train, y_train)

    print(X_train_filled)