from __future__ import annotations

import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import UploadFile

from .audio_pipeline import get_audio_duration, normalize_audio
from .config import DEFAULT_SAMPLE_RATE
from .demucs_infer import separate_guitar
from .exporters import artifact_extension, export_gp5, export_midi, export_musicxml, export_pdf
from .models import (
    Artifact,
    ArtifactKind,
    BranchAnalysis,
    ConfidenceSummary,
    JobOptions,
    JobResult,
    JobStatus,
    LowConfidenceSpan,
    PreprocessFacts,
    TranscriptionJob,
    SeparationMode,
    utc_now,
)
from .score_pipeline import build_score_part
from .storage import artifact_dir, save_job, upload_dir
from .tab_generator import generate_tab_events
from .transcription import choose_best_branch, fuse_branch_events, should_attempt_separation, transcribe_audio


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, TranscriptionJob] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)

    def create_job(self, uploaded_file: UploadFile, options: JobOptions) -> TranscriptionJob:
        job_id = uuid.uuid4().hex
        destination_dir = upload_dir(job_id)
        input_path = destination_dir / uploaded_file.filename
        with input_path.open("wb") as handle:
            shutil.copyfileobj(uploaded_file.file, handle)

        job = TranscriptionJob(
            id=job_id,
            filename=uploaded_file.filename,
            options=options,
            input_path=str(input_path),
        )
        self.jobs[job_id] = job
        save_job(job)
        self.executor.submit(self._run_job, job_id)
        return job

    def get_job(self, job_id: str) -> TranscriptionJob | None:
        return self.jobs.get(job_id)

    def _update(self, job: TranscriptionJob, status: JobStatus, progress: float, message: str) -> None:
        job.status = status
        job.progress = progress
        job.message = message
        job.updated_at = utc_now()
        save_job(job)

    def _run_job(self, job_id: str) -> None:
        job = self.jobs[job_id]
        warnings: list[str] = []
        try:
            self._update(job, JobStatus.preprocessing, 0.1, "Normalizing audio")
            normalized_path = upload_dir(job_id) / "normalized.wav"
            normalize_audio(Path(job.input_path), normalized_path, DEFAULT_SAMPLE_RATE)
            duration_sec = get_audio_duration(normalized_path)

            preprocess = PreprocessFacts(
                original_filename=job.filename,
                normalized_path=str(normalized_path),
                duration_sec=duration_sec,
                sample_rate=DEFAULT_SAMPLE_RATE,
            )

            branches = []
            self._update(job, JobStatus.transcribing, 0.35, "Transcribing original mix")
            branches.append(transcribe_audio(str(normalized_path), "original", job.options))

            wants_separation, separation_reason = should_attempt_separation(str(normalized_path), job.options)
            if separation_reason:
                warnings.append(separation_reason)
            if wants_separation:
                self._update(job, JobStatus.separating, 0.5, "Separating guitar stem")
                try:
                    stem_path = Path(separate_guitar(str(normalized_path), str(upload_dir(job_id) / "stems")))
                    preprocess.used_separation = True
                    preprocess.guitar_stem_path = str(stem_path)
                    self._update(job, JobStatus.transcribing, 0.65, "Transcribing isolated guitar stem")
                    branches.append(transcribe_audio(str(stem_path), "guitar_stem", job.options))
                except Exception as exc:
                    warnings.append(f"Guitar stem separation skipped: {exc}")

            chosen_branch = choose_best_branch(branches)
            warnings.extend(chosen_branch.warnings)
            cleanup_warnings = list(chosen_branch.cleanup_warnings)
            branch_scores = [
                BranchAnalysis(
                    branch_name=branch.branch_name,
                    raw_note_count=len(branch.raw_note_events),
                    cleaned_note_count=len(branch.note_events),
                    used_fallback=branch.used_fallback,
                    score=branch.score,
                )
                for branch in branches
            ]
            note_events = list(chosen_branch.note_events)
            raw_note_events = list(chosen_branch.raw_note_events)
            fusion_used = False

            if len(branches) > 1 and job.options.transcription_profile.value == "accurate":
                alternate = max(
                    (branch for branch in branches if branch.branch_name != chosen_branch.branch_name),
                    key=lambda branch: branch.score.overall,
                    default=None,
                )
                if alternate is not None:
                    note_events, fusion_used = fuse_branch_events(chosen_branch, alternate, job.options)
                    if fusion_used:
                        warnings.append(
                            f"Merged note candidates from {chosen_branch.branch_name} and {alternate.branch_name}."
                        )

            self._update(job, JobStatus.tabulating, 0.8, "Solving guitar tablature")
            tab_events = generate_tab_events(note_events, job.options)
            score_part, engraving_warnings = build_score_part(
                title=Path(job.filename).stem,
                tab_events=tab_events,
                audio_path=str(normalized_path),
                duration_sec=duration_sec,
                options=job.options,
            )

            confidence_summary = ConfidenceSummary(
                average_confidence=sum(event.confidence for event in note_events) / len(note_events)
                if note_events
                else 0.0,
                note_count=len(note_events),
                low_confidence_count=sum(1 for event in note_events if event.confidence < 0.45),
                branch_used=chosen_branch.branch_name,
            )
            low_confidence_spans = build_low_confidence_spans(note_events)

            self._update(job, JobStatus.exporting, 0.92, "Building artifacts")
            artifacts = self._build_artifacts(
                job_id=job.id,
                title=Path(job.filename).stem,
                note_events=note_events,
                tab_events=tab_events,
                score_part=score_part,
                summary=confidence_summary,
                requested_formats=job.options.export_formats,
                warnings=warnings,
            )

            job.result = JobResult(
                preprocess=preprocess,
                raw_note_events=raw_note_events,
                note_events=note_events,
                tab_events=tab_events,
                score_part=score_part,
                warnings=warnings,
                cleanup_warnings=cleanup_warnings,
                confidence_summary=confidence_summary,
                low_confidence_spans=low_confidence_spans,
                inferred_tempo_bpm=score_part.beat_grid.bpm if score_part else None,
                inferred_time_signature=score_part.time_signature
                if score_part
                else job.options.time_signature_override
                or job.options.time_signature_hint
                or "4/4",
                measure_count=len(score_part.measures) if score_part else 0,
                beat_confidence=score_part.beat_grid.confidence if score_part else None,
                engraving_warnings=engraving_warnings,
                engraved_output=score_part is not None and any(
                    artifact.kind == ArtifactKind.musicxml for artifact in artifacts
                ),
                branch_scores=branch_scores,
                note_level_fusion_used=fusion_used,
                artifacts=artifacts,
            )
            self._update(job, JobStatus.completed, 1.0, "Completed")
        except Exception as exc:
            job.error = str(exc)
            self._update(job, JobStatus.failed, 1.0, "Failed")

    def _build_artifacts(
        self,
        job_id: str,
        title: str,
        note_events,
        tab_events,
        score_part,
        summary: ConfidenceSummary,
        requested_formats: list[ArtifactKind],
        warnings: list[str],
    ) -> list[Artifact]:
        root = artifact_dir(job_id)
        artifacts: list[Artifact] = []
        for kind in requested_formats:
            output_path = root / f"transcription{artifact_extension(kind)}"
            try:
                if kind == ArtifactKind.midi:
                    export_midi(note_events, output_path)
                elif kind == ArtifactKind.guitar_pro:
                    export_gp5(tab_events, output_path, title)
                elif kind == ArtifactKind.pdf:
                    export_pdf(tab_events, summary, output_path)
                elif kind == ArtifactKind.musicxml:
                    if score_part is None:
                        raise RuntimeError("Engraved score is unavailable for MusicXML export.")
                    export_musicxml(score_part, output_path)
                artifacts.append(
                    Artifact(
                        kind=kind,
                        path=str(output_path),
                        download_url=f"/jobs/{job_id}/artifacts/{kind.value}",
                    )
                )
            except Exception as exc:
                warnings.append(f"Could not create {kind.value}: {exc}")
        return artifacts


def build_low_confidence_spans(note_events) -> list[LowConfidenceSpan]:
    spans: list[LowConfidenceSpan] = []
    pending = []
    for event in note_events:
        if event.confidence < 0.45:
            pending.append(event)
            continue
        if pending:
            spans.append(
                LowConfidenceSpan(
                    start_sec=pending[0].onset_sec,
                    end_sec=pending[-1].offset_sec,
                    average_confidence=sum(item.confidence for item in pending) / len(pending),
                )
            )
            pending = []
    if pending:
        spans.append(
            LowConfidenceSpan(
                start_sec=pending[0].onset_sec,
                end_sec=pending[-1].offset_sec,
                average_confidence=sum(item.confidence for item in pending) / len(pending),
            )
        )
    return spans


job_manager = JobManager()
