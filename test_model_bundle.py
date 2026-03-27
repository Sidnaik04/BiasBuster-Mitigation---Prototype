from app.db.database import SessionLocal
from app.db.models import UploadRecord
from app.core.model_loader import load_model

db = SessionLocal()
record = db.query(UploadRecord).filter_by(id=72).first()
if not record:
    # Just grab the latest one
    record = db.query(UploadRecord).order_by(UploadRecord.id.desc()).first()

print(f"Checking record ID: {record.id}")

model, preprocessor = load_model(record.model_path)
print(f"Model: {model}")
print(f"Preprocessor: {preprocessor}")
if preprocessor:
    if hasattr(preprocessor, 'transformers_'):
        print(f"Transformers: {preprocessor.transformers_}")
        
db.close()
