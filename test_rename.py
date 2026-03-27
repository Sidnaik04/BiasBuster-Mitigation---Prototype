import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LogisticRegression

def align_features(X: pd.DataFrame, model) -> pd.DataFrame:
    expected = None
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    elif hasattr(model, "estimator_") and hasattr(model.estimator_, "feature_names_in_"):
        expected = list(model.estimator_.feature_names_in_)
        
    if not expected:
        return X
        
    if set(X.columns) == set(expected):
        return X[expected]
        
    def clean(x):
        return re.sub(r'[^a-zA-Z0-9]', '_', str(x))
        
    cleaned_expected = {clean(x): x for x in expected}
    print("Cleaned Expected:", cleaned_expected)
    
    new_cols = []
    for c in X.columns:
        cl = clean(c)
        if cl in cleaned_expected:
            new_cols.append(cleaned_expected[cl])
        else:
            new_cols.append(c)
            
    X = X.copy()
    X.columns = new_cols
    print("New columns:", X.columns.tolist())
    
    available = [c for c in expected if c in X.columns]
    if len(available) == len(expected):
        return X[available]
    return X


class DummyModel:
    pass

model = DummyModel()
model.feature_names_in_ = np.array(['Unnamed:_0', 'Credit_amount', 'Saving_accounts'])

df = pd.DataFrame({'Unnamed: 0': [1], 'Credit amount': [2], 'Saving accounts': [3], 'Other': [4]})
aligned_df = align_features(df, model)
print("Final Columns:", aligned_df.columns.tolist())
