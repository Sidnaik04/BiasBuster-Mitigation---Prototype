import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from app.db.database import SessionLocal
from app.db.models import UploadRecord

db = SessionLocal()
record = db.query(UploadRecord).filter_by(id=72).first()

if not record:
    # grab most recent
    record = db.query(UploadRecord).order_by(UploadRecord.id.desc()).first()

print(f"Fixing model for Upload ID {record.id}")
print(f"Dataset path: {record.dataset_path}")
print(f"Target column: {record.target_column}")

df = pd.read_csv(record.dataset_path)
target = record.target_column

if target not in df.columns:
    # fallback target for Adult dataset
    target = 'income'

X = df.drop(columns=[target])
y = df[target]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

ml_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

print("Training new pipeline...")
ml_pipeline.fit(X, y)

# Overwrite the old model with the new pure pipeline
import os
os.makedirs(os.path.dirname(record.model_path), exist_ok=True)
joblib.dump(ml_pipeline, record.model_path)
print(f"Successfully saved pipeline to {record.model_path}")

db.close()
