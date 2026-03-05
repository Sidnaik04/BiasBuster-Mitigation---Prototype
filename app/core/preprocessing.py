import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


def preprocess_dataset(df: pd.DataFrame, target: str, sensitive: str):

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    if sensitive not in df.columns:
        raise ValueError(f"Sensitive attribute '{sensitive}' not found")

    df = df.copy()

    # -----------------------------
    # Target encoding
    # -----------------------------
    df[target] = df[target].astype(str).str.strip().str.lower()

    binary_map = {
        "<=50k": 0,
        "<=50k.": 0,
        ">50k": 1,
        ">50k.": 1,
        "no": 0,
        "yes": 1,
        "false": 0,
        "true": 1,
        "0": 0,
        "1": 1,
    }

    unique_vals = set(df[target].unique())

    if unique_vals.issubset(binary_map.keys()):
        df[target] = df[target].map(binary_map)
    else:
        df[target] = pd.factorize(df[target])[0]

    # -----------------------------
    # Sensitive attribute
    # -----------------------------
    df[sensitive] = df[sensitive].astype(str).str.strip()

    # -----------------------------
    # Feature encoding
    # -----------------------------
    X = df.drop(columns=[target])

    for col in X.columns:
        if is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
            continue

        X[col] = X[col].astype("string").fillna("__missing__").str.strip().astype(str)
        X[col] = pd.Categorical(X[col]).codes.astype(np.int64)

    y = df[target]
    sensitive_series = df[sensitive]

    return X, y, sensitive_series
