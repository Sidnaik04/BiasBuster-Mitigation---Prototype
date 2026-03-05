from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import UploadRecord
from app.core.dataset_loader import load_dataset
from app.core.model_loader import load_model
from app.core.preprocessing import preprocess_dataset
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
    model = load_model(record.model_path)
    raw_X = df.drop(columns=[target_column])

    X, y_true, sensitive = preprocess_dataset(df, target_column, sensitive_attribute)

    try:
        y_pred = model.predict(X)
    except Exception as first_exc:
        try:
            y_pred = model.predict(raw_X)
        except Exception as second_exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Model prediction failed on both preprocessed and raw features. "
                    f"preprocessed_error={first_exc}; raw_error={second_exc}"
                ),
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
