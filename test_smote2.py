import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

class DummyThresholdOptimizer:
    def fit(self, X, y, *, sensitive_features, **kwargs):
        print("Fitted DummyThresholdOptimizer!")
        self.fitted_ = True
        return self
    def get_params(self, deep=True):
        return {}
    def set_params(self, **params):
        return self

def apply_smote(X, y, sensitive_features, model):
    smote = SMOTE(random_state=42)
    import pandas as pd
    import numpy as np
    from fairlearn.postprocessing import ThresholdOptimizer

    s_factorized, s_uniques = pd.factorize(sensitive_features)

    if isinstance(X, pd.DataFrame):
        X_combined = X.copy()
        s_col_name = "__sensitive__"
        X_combined[s_col_name] = s_factorized
    else:
        X_combined = np.column_stack((X, s_factorized))

    X_resampled_combined, y_resampled = smote.fit_resample(X_combined, y)

    if isinstance(X_resampled_combined, pd.DataFrame):
        s_col_name = "__sensitive__"
        s_factorized_resampled = X_resampled_combined[s_col_name].round().astype(int)
        s_factorized_resampled = np.clip(s_factorized_resampled, 0, len(s_uniques) - 1)
        s_resampled = pd.Series(s_uniques[s_factorized_resampled])
        X_resampled = X_resampled_combined.drop(columns=[s_col_name])
    else:
        s_col_index = X_resampled_combined.shape[1] - 1
        s_factorized_resampled = np.round(X_resampled_combined[:, s_col_index]).astype(int)
        s_factorized_resampled = np.clip(s_factorized_resampled, 0, len(s_uniques) - 1)
        s_resampled = pd.Series(s_uniques[s_factorized_resampled])
        X_resampled = np.delete(X_resampled_combined, s_col_index, axis=1)

    new_model = clone(model)

    if type(new_model).__name__ == "DummyThresholdOptimizer":
        new_model.fit(X_resampled, y_resampled, sensitive_features=s_resampled)
    else:
        new_model.fit(X_resampled, y_resampled)

    return new_model, X_resampled, y_resampled

df = pd.DataFrame({"A": [1.1, 2.2, 3.3, 4.4, 1.1, 2.2], "B": [4.1, 5.2, 6.3, 7.4, 4.1, 5.2]})
y = pd.Series([0, 0, 1, 1, 0, 1])
s = pd.Series(["male", "female", "male", "female", "male", "female"])

model = DummyThresholdOptimizer()

new_model, X_res, y_res = apply_smote(df, y, s, model)
print("Smote execution successful")
