from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import UploadRecord
from app.core.dataset_loader import load_dataset
from app.core.model_loader import load_model
from app.core.preprocessing import preprocess_dataset, align_features
from app.fairness.evaluator import evaluate_baseline

router = APIRouter(prefix="/baseline", tags=["Baseline Evaluation"])


@router.post("/")
def run_baseline(
    upload_id: int,
    target_column: str,
    sensitive_attribute: str,
    db: Session = Depends(get_db),
):
    record = db.query(UploadRecord).filter_by(id=upload_id).first()

    if not record:
        raise HTTPException(status_code=404, detail="Upload record not found")

    df = load_dataset(record.dataset_path)
    model, preprocessor = load_model(record.model_path)
    raw_X = df.drop(columns=[target_column])

    X, y_true, sensitive = preprocess_dataset(df, target_column, sensitive_attribute)

    from sklearn.pipeline import Pipeline
    from fairlearn.postprocessing import ThresholdOptimizer

    # Fallback mapping for legacy models submitted purely as estimators (e.g. relying previously on pd.get_dummies)
    if not isinstance(model, Pipeline) and not (isinstance(model, ThresholdOptimizer) and getattr(model, "estimator", None) and isinstance(model.estimator, Pipeline)):
        import pandas as pd
        X_encoded = pd.get_dummies(raw_X)
        from app.core.preprocessing import align_features
        X_original = align_features(X_encoded, getattr(model, "estimator", model))
    else:
        # Enforce evaluation directly against raw strings for formal internal pipelines
        X_original = raw_X.copy()

    try:
        if isinstance(model, ThresholdOptimizer):
            y_pred = model.predict(X_original, sensitive_features=sensitive)
        else:
            y_pred = model.predict(X_original)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline inference failed structurally: Ensure uploaded model contains internal ColumnTransformer pipeline mapping strings to estimators. Error details: {exc}"
        )

    result = evaluate_baseline(y_true, y_pred, sensitive)

    return {
        "status": "success",
        "upload_id": upload_id,
        "target": target_column,
        "sensitive_attribute": sensitive_attribute,
        "baseline_metrics": result,
        "next_step": "mitigation_recommendation",
    }
