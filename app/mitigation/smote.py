from imblearn.over_sampling import SMOTE
from sklearn.base import clone


def apply_smote(X, y, model):

    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(X, y)

    new_model = clone(model)
    new_model.fit(X_resampled, y_resampled)

    return new_model, X_resampled, y_resampled