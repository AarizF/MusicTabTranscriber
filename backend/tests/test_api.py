from __future__ import annotations

import io
import math
import os
import time
import wave

from fastapi.testclient import TestClient

from backend.src.main import app

os.environ["MUSIC_TAB_DISABLE_BASIC_PITCH"] = "1"


def make_test_tone() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        frames = bytearray()
        for index in range(16000):
            value = int(16000 * math.sin(2 * math.pi * 440 * index / 16000))
            frames.extend(int(value).to_bytes(2, byteorder="little", signed=True))
        wav_file.writeframes(bytes(frames))
    return buffer.getvalue()


def test_job_lifecycle_returns_result_payload() -> None:
    client = TestClient(app)
    response = client.post(
        "/jobs",
        files={"file": ("tone.wav", make_test_tone(), "audio/wav")},
        data={"separation_mode": "off", "export_formats": "pdf,musicxml"},
    )
    assert response.status_code == 200
    job_id = response.json()["id"]

    for _ in range(120):
        job_response = client.get(f"/jobs/{job_id}")
        payload = job_response.json()
        if payload["status"] in {"completed", "failed"}:
            break
        time.sleep(0.2)

    assert payload["status"] == "completed", payload
    result_response = client.get(f"/jobs/{job_id}/result")
    assert result_response.status_code == 200
    result_payload = result_response.json()
    assert "note_events" in result_payload
    assert "raw_note_events" in result_payload
    assert "artifacts" in result_payload
    assert "engraved_output" in result_payload
    assert "branch_scores" in result_payload
    assert "cleanup_warnings" in result_payload
    assert any(artifact["kind"] == "musicxml" for artifact in result_payload["artifacts"])
