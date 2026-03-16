from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import librosa

from .config import DEFAULT_SAMPLE_RATE


def normalize_audio(input_path: Path, output_path: Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if input_path.suffix.lower() == ".wav":
        shutil.copy2(input_path, output_path)
        return output_path

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(input_path),
                "-ac",
                "2",
                "-ar",
                str(sample_rate),
                str(output_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return output_path

    # Fallback keeps the pipeline usable even when ffmpeg is missing.
    raise RuntimeError("ffmpeg is required to normalize non-WAV uploads.")


def get_audio_duration(path: Path) -> float:
    return float(librosa.get_duration(path=str(path)))
