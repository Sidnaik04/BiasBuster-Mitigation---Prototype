import joblib
from pathlib import Path


ALLOWED_MODEL_EXTENSIONS = {".pkl", ".joblib"}


def load_model(path: str):
    model_path = Path(path)
    if model_path.suffix.lower() not in ALLOWED_MODEL_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_MODEL_EXTENSIONS))
        raise ValueError(
            f"Unsupported model file type: {model_path.suffix}. Allowed types: {allowed}"
        )

    try:
        return joblib.load(path)
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")
