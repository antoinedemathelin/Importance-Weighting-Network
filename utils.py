import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import Callback


INT_TYPES = [int, np.int32, np.int64, np.uint, np.uint32, np.uint64]


def remove_columns(X, y, min_unique=2, max_cat_perc=0.5, max_cat=1000, dropna_y=True):
    drop_ind = y.index[np.any(y.isna(), axis=1)]
    X = X.drop(drop_ind)
    y = y.drop(drop_ind)
    
    remove_cols = {}
    for c in X.columns:
        if X[c].dtype.name == "datetime64[ns]":
            X[c] = X[c].values.astype("float")/10**18
            print(X[c].dtype.name)
        if X[c].nunique() < min_unique:
            remove_cols[c] = X[c].nunique()
        if (is_object_dtype(X[c]) or X[c].dtype in INT_TYPES) and (X[c].nunique()/len(X) > max_cat_perc):
            remove_cols[c] = X[c].nunique()
        if is_object_dtype(X[c]) and X[c].nunique() > max_cat:
            remove_cols[c] = X[c].nunique()
    
    print("Removed columns: %s"%str(remove_cols))
    print("X shape: %s, y shape: %s"%(str(X.shape), str(y.shape)))
    print("")
    
    X = X.drop(list(remove_cols.keys()), axis=1)
    return X, y


def columns_classification(df):
    numerical_features, categorical_features, other_features = ([],[],[])

    for c in df.columns:
        # XXX Boolean columns classed in numerical features
        if is_numeric_dtype(df[c]):
            numerical_features.append(c)
        elif is_object_dtype(df[c]):
            categorical_features.append(c)
        else:
            other_features.append(c)

    return numerical_features, categorical_features, other_features


def select_columns(df, col_type=None, cols=None):
    """
    Select columns from a dataframe.
    """
    if cols is None:
        numerical_columns, categorical_columns = columns_classification(df)[:2]
        if col_type is None:
            return df
        elif col_type == 'numerical':
            return df[numerical_columns]
        elif col_type == 'categorical':
            return df[categorical_columns]
        else:
            raise TypeError(
                "Please provide a valid col_type: 'numerical' or 'categorical'")
    else:
        return df[cols]


class AdaptPipeline(Pipeline):
    
    def fit_transform(self, X, y=None, **fit_params):
        if len(X.shape) > 1 and X.shape[1] == 0:
            return X
        else:
            return super().fit_transform(X, y, **fit_params)
        
    def fit(self, X, y=None, **fit_params):
        if len(X.shape) > 1 and X.shape[1] == 0:
            return self
        else:
            return super().fit(X, y, **fit_params)
        
    def transform(self, X):
        if len(X.shape) > 1 and X.shape[1] == 0:
            return X
        else:
            return super().transform(X)
        
        
class CustomEarlyStopping(Callback):
    def __init__(self, min_delta=1e-5, patience=10, first_patience=1):
        super(Callback, self).__init__()
        self.min_delta = min_delta
        self.patience = patience
        self.first_patience = first_patience
        self.no_improvement = 0
        self.best = np.inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get("loss")
        if epoch >= self.first_patience:
            if current+self.min_delta < self.best:
                self.best = current
                self.no_improvement = 0
            else:
                self.no_improvement += 1
        if self.no_improvement >= self.patience:
            self.model.stop_training = True
            
            
class EarlyStoppingByLossVal(Callback):
    def __init__(self, value=1e-4):
        super(Callback, self).__init__()
        self.value = value

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get("loss")
        if current < self.value:
            self.model.stop_training = True
            
            
def get_sample_bias_X(X, a=3., b=8.):
    x = PCA(1).fit_transform(X).ravel()
    mean = x.min() + (x.mean() - x.min())/a
    std = (x.mean() - x.min())/b
    sample_bias = (1/(np.sqrt(2*np.pi)*std)) * np.exp(-(x-mean)**2 / (2*std**2))
    return sample_bias, x


def get_sample_bias_y(y, a=-3, b=3.):
    x = PCA(1).fit_transform(StandardScaler().fit_transform(y)).ravel()
    x = np.clip(a + b*x, -np.inf, 10)
    sample_bias = np.exp(x)/(1 + np.exp(x))
    return sample_bias, x
        

# Create the preprocessing pipelines for numerical data.
numerical_transformer = AdaptPipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean', verbose=1)),
        ('scaler', StandardScaler())])

# Create the preprocessing pipelines for categorical data.
categorical_transformer = AdaptPipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing',verbose=1)),
    ('onehot', OneHotEncoder(categories='auto',sparse=False,handle_unknown='ignore'))])

# Combine all elements of preprocessor
preprocessor = Pipeline(steps=[
        ("union", FeatureUnion(transformer_list=[
    ('numerical', Pipeline(steps=[
        ('get_num_cols', FunctionTransformer(select_columns, kw_args=dict(col_type="numerical"))),
        ('transformer', numerical_transformer)])),
    ('categorical', Pipeline(steps=[
        ('get_cat_cols', FunctionTransformer(select_columns, kw_args=dict(col_type="categorical"))),
        ('transformer',categorical_transformer)]))
    ]))
])