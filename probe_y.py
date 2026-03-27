import sys
import pandas as pd
from sqlalchemy.orm import Session
from app.db.database import get_db, SessionLocal
from app.db.models import UploadRecord
from app.core.preprocessing import preprocess_dataset

db = SessionLocal()
record = db.query(UploadRecord).filter_by(id=72).first()
if not record:
    print("Record not found")
    sys.exit(1)

df = pd.read_csv(record.dataset_path)

# check before preprocessing
print("Unique raw targets:", df["Test Results"].unique())
print("Null count:", df["Test Results"].isnull().sum())

# check after preprocessing
target_column = "Test Results"
df[target_column] = df[target_column].astype(str).str.strip().str.lower()
print("Unique lowered:", df[target_column].unique())

encoded = pd.factorize(df[target_column])[0]
print("Unique factorized (what y gets):", pd.Series(encoded).unique())

db.close()
