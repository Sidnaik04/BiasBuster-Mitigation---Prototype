import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer

def get_expected_features(model):
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    if hasattr(model, 'estimator_') and hasattr(model.estimator_, 'feature_names_in_'):
        return list(model.estimator_.feature_names_in_)
    return None

def standardize_columns(df, expected):
    if expected is None:
        return df
    
    df_new = df.copy()
    rename_mapping = {}
    for col in df_new.columns:
        if col not in expected:
            col_underscore = col.replace(" ", "_")
            if col_underscore in expected:
                rename_mapping[col] = col_underscore
            else:
                for exp_col in expected:
                    if col.lower().replace(" ", "_") == exp_col.lower().replace(" ", "_"):
                        rename_mapping[col] = exp_col
                        break
                        
    if rename_mapping:
        df_new = df_new.rename(columns=rename_mapping)
    return df_new

def align_ohe(model, df):
    expected = get_expected_features(model)
    if expected is not None:
        if list(df.columns) != expected:
            df_ohe = pd.get_dummies(df, dtype=float)
            missing = set(expected) - set(df_ohe.columns)
            for c in missing:
                df_ohe[c] = 0.0
            
            valid_cols = [c for c in expected if c in df_ohe.columns]
            return df_ohe[valid_cols]
    return df

def predict_with_fallback(model, X, raw_X, sensitive):
    errors = []
    
    expected = get_expected_features(model)
    if expected is not None:
        X = standardize_columns(X, expected)
        raw_X = standardize_columns(raw_X, expected)
    
    def try_pred(data):
        if isinstance(model, ThresholdOptimizer):
            return model.predict(data, sensitive_features=sensitive)
        return model.predict(data)
        
    try:
        return try_pred(X)
    except Exception as e:
        errors.append(f"preprocessed_error={e}")
        
    try:
        return try_pred(raw_X)
    except Exception as e:
        errors.append(f"raw_error={e}")
        
    try:
        X_aligned = align_ohe(model, raw_X)
        return try_pred(X_aligned)
    except Exception as e:
        errors.append(f"aligned_error={e}")
        
    raise ValueError("Model prediction failed on preprocessed, raw, and aligned features. " + "; ".join(errors))
