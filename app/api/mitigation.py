import os
import uuid
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
import pandas as pd


from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split

from app.db.database import get_db
from app.db.models import UploadRecord, MitigationRun

from app.core.dataset_loader import load_dataset
from app.core.model_loader import load_model
from app.core.preprocessing import preprocess_dataset
from app.core.persistence import get_latest_model

from app.fairness.evaluator import evaluate_baseline
from app.fairness.comparison import compare_metrics

from app.mitigation.recommender import recommend_strategy
from app.mitigation.smote import apply_smote
from app.mitigation.reweighting import compute_sample_weights
from app.mitigation.threshold import apply_threshold_optimizer
from fairlearn.postprocessing import ThresholdOptimizer
from app.config import settings
from sklearn.pipeline import Pipeline


router = APIRouter(prefix="/mitigation", tags=["Bias Mitigation"])


# =====================================================
# PHASE 3 — STRATEGY RECOMMENDATION
# =====================================================


@router.post("/recommend")
def recommend_mitigation(
    upload_id: int,
    baseline_metrics: dict,
    db: Session = Depends(get_db),
):
    record = db.query(UploadRecord).filter_by(id=upload_id).first()

    if not record:
        raise HTTPException(status_code=404, detail="Upload record not found")

    try:
        recommendation = recommend_strategy(baseline_metrics)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "status": "success",
        "upload_id": upload_id,
        "recommendation": recommendation,
        "next_step": (
            "apply_mitigation"
            if recommendation["recommended_strategy"] != "none"
            else "no_action_required"
        ),
    }


# =====================================================
# PHASE 4 — APPLY MITIGATION
# =====================================================


@router.post("/apply")
def apply_mitigation(
    upload_id: int,
    target_column: str,
    sensitive_attribute: str,
    strategy: str,
    strategy_config: dict = {},
    db: Session = Depends(get_db),
):

    record = db.query(UploadRecord).filter_by(id=upload_id).first()

    if not record:
        raise HTTPException(status_code=404, detail="Upload record not found")

    # -------------------------------------------------
    # Load dataset + model
    # -------------------------------------------------

    df = load_dataset(record.dataset_path)

    # model_path = get_latest_model(upload_id, db, record.model_path)
    model_path = record.model_path
    model = load_model(model_path)
    print(model)
  

    raw_X = df.drop(columns=[target_column])

    X, y, sensitive = preprocess_dataset(
        df,
        target_column,
        sensitive_attribute,
    )
    # Save original dataset for fairness evaluation
    X_original = X.copy()
    y_original = y.copy()
    sensitive_original = sensitive.copy()
    # Compute baseline metrics on original dataset

    try:
     if isinstance(model, ThresholdOptimizer):
        y_pred_base = model.predict(
            X_original,
            sensitive_features=sensitive_original
        )
     else:
        y_pred_base = model.predict(X_original)

    except Exception as e:
     raise HTTPException(
        status_code=400,
        detail=f"Model prediction failed: {str(e)}"
    )

    baseline_metrics = evaluate_baseline(
    y_original,
    y_pred_base,
    sensitive_original
)

    # -------------------------------------------------
    # Train/Test split
    # -------------------------------------------------

    # test_size = strategy_config.get("test_size", 0.2)

    # X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    #     X,
    #     y,
    #     sensitive,
    #     test_size=test_size,
    #     random_state=42,
    #     stratify=y,
    # )

    # mitigated_model = model
    
# Apply mitigation strategy
# -------------------------------------------------
    # Apply mitigation strategy
# -------------------------------------------------
    rows_before = None
    rows_after = None

# =========================
# SMOTE (FULL DATA)
# =========================
    if strategy == "smote":

    

     mitigated_model, X_balanced, y_balanced, sensitive_balanced = apply_smote(
        X, y, sensitive, model
    )

     rows_before = len(X)
     rows_after = len(X_balanced)

    # Save debiased dataset
     df_balanced = pd.DataFrame(X_balanced, columns=X.columns)
     df_balanced[target_column] = y_balanced

     dataset_id = uuid.uuid4().hex 
     dataset_path = os.path.join(
        settings.ARTIFACT_DIR,
        "datasets",
        f"debiased_{dataset_id}.csv"
    )

     os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
     df_balanced.to_csv(dataset_path, index=False)

     y_pred_after = mitigated_model.predict(X_balanced)

     after_metrics = evaluate_baseline(
        y_balanced,
        y_pred_after,
        sensitive_balanced
    )

# =========================
# REWEIGHTING (FULL DATA)
# =========================
    elif strategy == "reweighting":

    

     weights = compute_sample_weights(y, sensitive)

     mitigated_model = clone(model)

     if isinstance(mitigated_model, Pipeline):
        mitigated_model.fit(X, y, model__sample_weight=weights)
     else:
        mitigated_model.fit(X, y, sample_weight=weights)

     y_pred_after = mitigated_model.predict(X)

     after_metrics = evaluate_baseline(
        y,
        y_pred_after,
        sensitive
    )

# =========================
# THRESHOLD OPTIMIZER (FULL DATA)
# =========================
    elif strategy == "threshold":

      mitigated_model = apply_threshold_optimizer(
        model,
        X,
        y,
        sensitive,
        grid_size=strategy_config.get("grid_size", 200),
    )

      y_pred_after = mitigated_model.predict(
        X,
        sensitive_features=sensitive,
    )

      after_metrics = evaluate_baseline(
        y,
        y_pred_after,
        sensitive
    )

    else:
     raise HTTPException(status_code=400, detail="Unknown strategy")
  



    improvement_score = (
    baseline_metrics.get("bias_severity_score", 0)
    - after_metrics.get("bias_severity_score", 0)
)

    # -------------------------------------------------
    # Save mitigated model
    # -------------------------------------------------

    model_id = uuid.uuid4().hex

    model_path = os.path.join(
        settings.ARTIFACT_DIR,
        "models",
        f"mitigated_{model_id}.pkl",
    )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(mitigated_model, model_path)

    # -------------------------------------------------
    # Save mitigation run to DB
    # -------------------------------------------------

    run = MitigationRun(
    upload_id=upload_id,
    sensitive_attribute=sensitive_attribute,
    strategy=strategy,
    before_metrics=baseline_metrics,
    after_metrics=after_metrics,
    artifact_model_path=model_path,
)

    db.add(run)
    db.commit()
    db.refresh(run)

    comparison = compare_metrics(baseline_metrics, after_metrics)

    return {
    "status": "mitigation_success",
    "strategy": strategy,
    "rows_before": rows_before if strategy == "smote" else None,
    "rows_after": rows_after if strategy == "smote" else None,
    "improvement_score": improvement_score,
    "before": baseline_metrics,
    "after": after_metrics,
    "comparison": comparison,
    "artifact_model": model_path,
}
