import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

class DummyThresholdOptimizer:
    def fit(self, X, y, *, sensitive_features, **kwargs):
        print("Fitted ThresholdOptimizer with sensitive_features!")
        self.fitted_ = True
        return self

def apply_smote(X, y, sensitive_features, model):
    smote = SMOTE(random_state=42)
    
    # 1. Combine X and sensitive_features
    # We must ensure sensitive_features is correctly aligned
    if isinstance(X, pd.DataFrame):
        X_combined = X.copy()
        s_col_name = "__sensitive__"
        X_combined[s_col_name] = sensitive_features.values
    else:
        # numpy array
        X_combined = np.column_stack((X, sensitive_features))
        s_col_index = X_combined.shape[1] - 1
        
    X_resampled_combined, y_resampled = smote.fit_resample(X_combined, y)
    
    # 2. Separate
    if isinstance(X_resampled_combined, pd.DataFrame):
        s_resampled = X_resampled_combined[s_col_name].round().astype(int)
        X_resampled = X_resampled_combined.drop(columns=[s_col_name])
    else:
        s_resampled = np.round(X_resampled_combined[:, s_col_index]).astype(int)
        X_resampled = np.delete(X_resampled_combined, s_col_index, axis=1)
        
    new_model = clone(model)
    # Mocking behavior
    if type(new_model).__name__ == "DummyThresholdOptimizer":
        new_model.fit(X_resampled, y_resampled, sensitive_features=s_resampled)
    else:
        new_model.fit(X_resampled, y_resampled)
        
    return new_model, X_resampled, y_resampled

df = pd.DataFrame({"A": [1, 2, 3, 4, 1, 2], "B": [4, 5, 6, 7, 4, 5]})
y = pd.Series([0, 0, 1, 1, 0, 1])
s = pd.Series([1, 0, 1, 0, 1, 0])

model = DummyThresholdOptimizer()

new_model, X_res, y_res = apply_smote(df, y, s, model)
print("Smote Test successful")
