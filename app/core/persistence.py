def get_latest_model(upload_id, db, base_model_path):
    """
    Returns the most recent mitigated model if available,
    otherwise returns the original uploaded model.
    """

    from app.db.models import MitigationRun

    latest = (
        db.query(MitigationRun)
        .filter(MitigationRun.upload_id == upload_id)
        .order_by(MitigationRun.created_at.desc())
        .first()
    )

    if latest:
        return latest.artifact_model_path

    return base_model_path