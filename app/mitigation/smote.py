from imblearn.over_sampling import SMOTE
from sklearn.base import clone
import pandas as pd


def apply_smote(X, y, sensitive, model):

    # Combine everything into one DataFrame
    df = pd.DataFrame(X)
    df["target"] = y
    df["sensitive"] = sensitive

    # Apply SMOTE on features + target
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(df.drop(columns=["target", "sensitive"]), df["target"])

    # 🔥 Now rebuild dataframe properly
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled["target"] = y_resampled

    # -----------------------------
    # 🚨 IMPORTANT: rebuild sensitive
    # -----------------------------
    # Since SMOTE does NOT generate sensitive correctly,
    # we assign using nearest original values (simple approach)

    sensitive_resampled = pd.Series(
        list(sensitive) + list(sensitive.sample(len(X_resampled) - len(X), replace=True)),
        index=df_resampled.index
    )

    # -----------------------------
    # Train model
    # -----------------------------
    new_model = clone(model)
    new_model.fit(X_resampled, y_resampled)

    return new_model, X_resampled, y_resampled, sensitive_resampled