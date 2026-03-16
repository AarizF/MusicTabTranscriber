import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
APP_DATA_DIR = PROJECT_ROOT / "app_data"
UPLOADS_DIR = APP_DATA_DIR / "uploads"
JOBS_DIR = APP_DATA_DIR / "jobs"
ARTIFACTS_DIR = APP_DATA_DIR / "artifacts"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_MAX_FRET = 20
DEFAULT_EXPORT_FORMATS = ("mid", "gp5", "pdf", "musicxml")
MPL_CONFIG_DIR = APP_DATA_DIR / ".matplotlib"

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))


def ensure_app_dirs() -> None:
    for path in (APP_DATA_DIR, UPLOADS_DIR, JOBS_DIR, ARTIFACTS_DIR, MPL_CONFIG_DIR):
        path.mkdir(parents=True, exist_ok=True)
