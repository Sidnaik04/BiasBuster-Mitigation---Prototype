import os
import uuid
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import UploadRecord
from app.core.dataset_loader import load_dataset
from app.config import settings

router = APIRouter(prefix="/upload", tags=["Upload"])

DATA_DIR = os.path.join(settings.ARTIFACT_DIR, "datasets")
MODEL_DIR = os.path.join(settings.ARTIFACT_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


@router.post("/")
def upload_dataset_and_model(
    dataset: UploadFile = File(...),
    model: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not dataset.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Dataset must be CSV")

    if not model.filename.endswith((".pkl", ".joblib")):
        raise HTTPException(status_code=400, detail="Model must be pkl or joblib")

    ds_name = f"{uuid.uuid4().hex}_{dataset.filename}"
    md_name = f"{uuid.uuid4().hex}_{model.filename}"

    ds_path = os.path.join(DATA_DIR, ds_name)
    md_path = os.path.join(MODEL_DIR, md_name)

    with open(ds_path, "wb") as f:
        f.write(dataset.file.read())

    with open(md_path, "wb") as f:
        f.write(model.file.read())

    df = load_dataset(ds_path)

    record = UploadRecord(
        dataset_path=ds_path,
        model_path=md_path,
        dataset_columns=df.columns.tolist(),
        dataset_rows=len(df),
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "status": "success",
        "upload_id": record.id,
        "dataset_info": {
            "rows": record.dataset_rows,
            "columns": record.dataset_columns,
        },
        "next_step": "select_target_and_sensitive_attribute",
    }
