from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from .config import DEFAULT_EXPORT_FORMATS, DEFAULT_MAX_FRET


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    queued = "queued"
    preprocessing = "preprocessing"
    separating = "separating"
    transcribing = "transcribing"
    tabulating = "tabulating"
    exporting = "exporting"
    completed = "completed"
    failed = "failed"


class SeparationMode(str, Enum):
    off = "off"
    demucs = "demucs"
    auto = "auto"


class TranscriptionProfile(str, Enum):
    fast = "fast"
    accurate = "accurate"


class ArtifactKind(str, Enum):
    midi = "mid"
    guitar_pro = "gp5"
    pdf = "pdf"
    musicxml = "musicxml"


class QuantizationLevel(str, Enum):
    sixteenth = "16th"


class EngravedLayout(str, Enum):
    linked_notation_tab = "linked_notation_tab"


class TechniqueHint(str, Enum):
    picked = "picked"
    bend = "bend"
    slide = "slide"
    legato = "legato"


class JobOptions(BaseModel):
    transcription_profile: TranscriptionProfile = TranscriptionProfile.accurate
    separation_mode: SeparationMode = SeparationMode.auto
    tuning: list[int] = Field(default_factory=lambda: [40, 45, 50, 55, 59, 64])
    max_fret: int = DEFAULT_MAX_FRET
    prefer_lower_positions: bool = True
    lead_guitar_mode: bool = True
    tempo_override_bpm: float | None = None
    tempo_hint_bpm: float | None = None
    time_signature_override: str | None = None
    time_signature_hint: str | None = None
    quantization_level: QuantizationLevel = QuantizationLevel.sixteenth
    engraved_layout: EngravedLayout = EngravedLayout.linked_notation_tab
    export_formats: list[ArtifactKind] = Field(
        default_factory=lambda: [ArtifactKind(value) for value in DEFAULT_EXPORT_FORMATS]
    )


class Artifact(BaseModel):
    kind: ArtifactKind
    path: str
    download_url: str
    created_at: datetime = Field(default_factory=utc_now)


class NoteEvent(BaseModel):
    onset_sec: float
    offset_sec: float
    midi_pitch: int
    confidence: float = 0.0
    pitch_bend_cents: float = 0.0
    technique_hint: TechniqueHint = TechniqueHint.picked
    source_branch: Literal["original", "guitar_stem"] = "original"


class TabEvent(BaseModel):
    onset_sec: float
    offset_sec: float
    midi_pitch: int
    string: int
    fret: int
    confidence: float = 0.0
    pitch_bend_cents: float = 0.0
    technique_hint: TechniqueHint = TechniqueHint.picked
    source_branch: Literal["original", "guitar_stem"] = "original"


class BeatGrid(BaseModel):
    bpm: float
    beat_times: list[float] = Field(default_factory=list)
    beat_duration_sec: float
    subdivision_duration_sec: float
    beats_per_measure: int = 4
    beat_unit: int = 4
    confidence: float = 0.0


class QuantizedNote(BaseModel):
    onset_slot: int
    duration_slots: int
    midi_pitch: int
    string: int | None = None
    fret: int | None = None
    confidence: float = 0.0
    pitch_bend_cents: float = 0.0
    technique_hint: TechniqueHint = TechniqueHint.picked
    tie_start: bool = False
    tie_stop: bool = False
    is_chord_tone: bool = False


class ScoreEvent(BaseModel):
    voice: int = 1
    start_slot: int
    duration_slots: int
    note_type: str
    dots: int = 0
    is_rest: bool = False
    notes: list[QuantizedNote] = Field(default_factory=list)


class ScoreMeasure(BaseModel):
    number: int
    total_slots: int
    voice_count: int = 1
    events: list[ScoreEvent] = Field(default_factory=list)


class ScorePart(BaseModel):
    title: str
    time_signature: str
    divisions: int
    beat_grid: BeatGrid
    measures: list[ScoreMeasure] = Field(default_factory=list)
    quantization_error_sec: float = 0.0
    measure_fill_valid: bool = False


class ScoreArtifact(BaseModel):
    format: Literal["musicxml"] = "musicxml"
    path: str
    download_url: str


class BranchScoreBreakdown(BaseModel):
    average_confidence: float = 0.0
    onset_alignment: float = 0.0
    playable_ratio: float = 0.0
    density_score: float = 0.0
    duplicate_penalty: float = 0.0
    octave_penalty: float = 0.0
    overall: float = 0.0


class BranchAnalysis(BaseModel):
    branch_name: Literal["original", "guitar_stem"]
    raw_note_count: int = 0
    cleaned_note_count: int = 0
    used_fallback: bool = False
    score: BranchScoreBreakdown


class ConfidenceSummary(BaseModel):
    average_confidence: float = 0.0
    note_count: int = 0
    low_confidence_count: int = 0
    branch_used: Literal["original", "guitar_stem"] = "original"


class LowConfidenceSpan(BaseModel):
    start_sec: float
    end_sec: float
    average_confidence: float


class PreprocessFacts(BaseModel):
    original_filename: str
    normalized_path: str
    duration_sec: float
    sample_rate: int
    used_separation: bool = False
    guitar_stem_path: str | None = None


class JobResult(BaseModel):
    preprocess: PreprocessFacts
    raw_note_events: list[NoteEvent] = Field(default_factory=list)
    note_events: list[NoteEvent]
    tab_events: list[TabEvent]
    score_part: ScorePart | None = None
    warnings: list[str] = Field(default_factory=list)
    cleanup_warnings: list[str] = Field(default_factory=list)
    confidence_summary: ConfidenceSummary
    low_confidence_spans: list[LowConfidenceSpan] = Field(default_factory=list)
    inferred_tempo_bpm: float | None = None
    inferred_time_signature: str | None = None
    measure_count: int = 0
    beat_confidence: float | None = None
    engraving_warnings: list[str] = Field(default_factory=list)
    engraved_output: bool = False
    branch_scores: list[BranchAnalysis] = Field(default_factory=list)
    note_level_fusion_used: bool = False
    artifacts: list[Artifact] = Field(default_factory=list)


class TranscriptionJob(BaseModel):
    id: str
    status: JobStatus = JobStatus.queued
    progress: float = 0.0
    message: str = "Queued"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    filename: str
    options: JobOptions
    input_path: str
    result: JobResult | None = None
    error: str | None = None
