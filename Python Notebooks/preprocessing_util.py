import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder

log_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'YEARS_EMPLOYED', 'YEARS_LAST_PHONE_CHANGE', 'YEARS_REGISTRATION']

def get_column_groups(X):
    
    numeric_cols = [col for col in X.columns if X[col].dtype != 'object' and X[col].nunique() > 2]
    scale_only_cols = list(set(numeric_cols) - set(log_cols))
    binary_cols = [col for col in X.columns if X[col].dtype != 'object' and X[col].nunique() == 2]
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']

    return {
        'log_col': log_cols,
        'scale_only_cols': scale_only_cols,
        'binary_cols': binary_cols,
        'categorical_cols': categorical_cols
    }

log_pipeline = Pipeline([
    ('log', FunctionTransformer(np.log1p, validate=True)),
    ('scale', StandardScaler())
])

binary_passthrough = Pipeline([
    ('identity', FunctionTransformer(validate=True))
])

scale_only_pipeline = Pipeline([
    ('scale', StandardScaler())
])


def get_transformers(log_col, scale_only_cols, categorical_cols, binary_cols):
    return [
        ('log_scale_pipeline', log_pipeline, log_col),
         ('binary_passthrough', binary_passthrough, binary_cols),
        ('scale_only', scale_only_pipeline, scale_only_cols),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categorical_cols)
    ]
