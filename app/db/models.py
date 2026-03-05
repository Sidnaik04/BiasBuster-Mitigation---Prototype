from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.sql import func
from app.db.database import Base


class UploadRecord(Base):
    __tablename__ = "upload_records"

    id = Column(Integer, primary_key=True, index=True)
    
    dataset_path = Column(String, nullable=False)
    model_path = Column(String, nullable=False)
    
    dataset_columns = Column(JSON)
    dataset_rows = Column(JSON)

    created_at = Column(DateTime(timezone=True), server_default=func.now())



class MitigationRun(Base):
    __tablename__ = "mitigation_runs"

    id = Column(Integer, primary_key=True, index=True)

    upload_id = Column(Integer, nullable=False)
    sensitive_attribute = Column(String, nullable=False)
    strategy = Column(String, nullable=False)

    before_metrics = Column(JSON)
    after_metrics = Column(JSON)

    artifact_model_path = Column(String)
    artifact_dataset_path = Column(String)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
