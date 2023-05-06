import pandas as pd
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self,missing_threshold = 0.9, imputation_method = 'KNN', n_neighbors = 5, weights = 'uniform'):
        self.missing_threshold = missing_threshold
        self.imputation_method = imputation_method

        if self.imputation_method not in ['xgboost', 'KNN']:
            raise ValueError(f"Invalid imputation_method: {self.imputation_method}. Supported values: 'xgboost', 'KNN'")
        
        self.n_neighbors = n_neighbors
        self.weights = weights
        
        self.columns_to_drop_ = None
        self.columns_with_NaN = None
        self.imputer_ = None

    def fit(self,X,y = None):
        self.columns_to_drop_ = X.columns[X.isnull().mean() > self.missing_threshold].to_list()
        X.drop(columns = self.columns_to_drop_, axis = 1, inplace = True)
        
        if self.imputation_method == 'xgboost':
           self.columns_with_NaN = X.columns[X.isnull().any()].tolist()
           for col in self.columns_with_NaN:
            fit_col = self.__fit_xgb_model(X,y,col)

            if self.imputer_ is None:
                 self.imputer_ = [fit_col]
            else:
                self.imputer_.append(fit_col) 
                 
        elif self.imputation_method == 'KNN':
            self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights)
            self.imputer_.fit(X)

        return self

    
    def transform(self,X):
        
        if all(col in X.columns for col in self.columns_to_drop_):
            X.drop(columns=self.columns_to_drop_, axis = 1, inplace = True)

        if self.imputation_method == 'xgboost':

            for i, col in enumerate(self.columns_with_NaN):
                X = self.__transform_with_xgb(self.imputer_[i], X,col)
        
        elif self.imputation_method == 'KNN':
            X = pd.DataFrame(self.imputer_.transform(X), columns = X.columns)
        
        return X
    
    def fit_transform(self,X,y = None):
        self.fit(X,y)
        return self.transform(X)

    def __fit_xgb_model(X_train, y_train, target_col):


        nan_ix = X_train[X_train[target_col].isnull()].index
        
        X_train = X_train.drop(nan_ix, axis = 0)
        y_train = y_train.drop(nan_ix, axis = 0)

        # Train the model
        model = XGBRegressor()
        model.fit(X_train, y_train)

        return model
    
    def __transform_with_xgb(model, X_test, target_col):
        # Get the indices of the rows with missing values
        nan_ix = X_test[X_test[target_col].isnull()].index

        # Create a copy of the test data to store the transformed values
        X_test_transformed = X_test.copy()

        # Get the test data without missing values
        # test = transformed_df[~transformed_df.index.isin(nan_ix)]
        test = X_test[X_test.index.isin(nan_ix)]

        # Use the trained model to predict the missing values
        pred = model.predict(test)

        # Update the missing values in the transformed_df
        X_test_transformed.loc[nan_ix, target_col] = pred

        return X_test_transformed
    

if __name__ == '__main__':
    df = pd.read_csv('hospital_deaths_train.csv')
    target_variable = 'In-hospital_death'
    
    X = df.drop(['recordid',target_variable], axis = 1)
    y = df[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=42, stratify = y)

    preprocessor = Preprocessor()

    # print(X.columns[X.isnull().mean() > 0.9].to_list())

    X_train_filled = preprocessor.fit_transform(X_train, y_train)

    # print(X_test.columns == X_train.columns)

    print(X_train_filled.head())