from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .job_manager import job_manager
from .models import (
    ArtifactKind,
    EngravedLayout,
    JobOptions,
    JobStatus,
    QuantizationLevel,
    SeparationMode,
    TranscriptionProfile,
)


app = FastAPI(title="MusicTabTranscriber API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "service": "MusicTabTranscriber"}


@app.post("/jobs")
async def create_job(
    file: UploadFile = File(...),
    transcription_profile: TranscriptionProfile = Form(TranscriptionProfile.accurate),
    separation_mode: SeparationMode = Form(SeparationMode.auto),
    max_fret: int = Form(20),
    prefer_lower_positions: bool = Form(True),
    lead_guitar_mode: bool = Form(True),
    tempo_override_bpm: float | None = Form(None),
    tempo_hint_bpm: float | None = Form(None),
    time_signature_override: str | None = Form(None),
    time_signature_hint: str | None = Form(None),
    quantization_level: QuantizationLevel = Form(QuantizationLevel.sixteenth),
    engraved_layout: EngravedLayout = Form(EngravedLayout.linked_notation_tab),
    export_formats: str = Form("mid,gp5,pdf,musicxml"),
):
    formats = [ArtifactKind(value.strip()) for value in export_formats.split(",") if value.strip()]
    options = JobOptions(
        transcription_profile=transcription_profile,
        separation_mode=separation_mode,
        max_fret=max_fret,
        prefer_lower_positions=prefer_lower_positions,
        lead_guitar_mode=lead_guitar_mode,
        tempo_override_bpm=tempo_override_bpm,
        tempo_hint_bpm=tempo_hint_bpm,
        time_signature_override=time_signature_override,
        time_signature_hint=time_signature_hint,
        quantization_level=quantization_level,
        engraved_layout=engraved_layout,
        export_formats=formats,
    )
    job = job_manager.create_job(file, options)
    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/jobs/{job_id}/result")
def get_result(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.completed or not job.result:
        raise HTTPException(status_code=409, detail="Job is not completed yet")
    return job.result


@app.get("/jobs/{job_id}/artifacts/{kind}")
def get_artifact(job_id: str, kind: ArtifactKind):
    job = job_manager.get_job(job_id)
    if not job or not job.result:
        raise HTTPException(status_code=404, detail="Artifact not found")

    for artifact in job.result.artifacts:
        if artifact.kind == kind:
            return FileResponse(artifact.path, filename=f"{job.filename.rsplit('.', 1)[0]}.{kind.value}")
    raise HTTPException(status_code=404, detail="Artifact not found")
