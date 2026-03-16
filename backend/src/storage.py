from __future__ import annotations

import json
from pathlib import Path

from .config import ARTIFACTS_DIR, JOBS_DIR, UPLOADS_DIR, ensure_app_dirs
from .models import TranscriptionJob


ensure_app_dirs()


def job_dir(job_id: str) -> Path:
    path = JOBS_DIR / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def upload_dir(job_id: str) -> Path:
    path = UPLOADS_DIR / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_dir(job_id: str) -> Path:
    path = ARTIFACTS_DIR / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_job(job: TranscriptionJob) -> None:
    path = job_dir(job.id) / "job.json"
    path.write_text(job.model_dump_json(indent=2), encoding="utf-8")


def load_job(job_id: str) -> TranscriptionJob | None:
    path = job_dir(job_id) / "job.json"
    if not path.exists():
        return None
    return TranscriptionJob.model_validate(json.loads(path.read_text(encoding="utf-8")))
