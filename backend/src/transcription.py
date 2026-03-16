from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import builtins
import importlib.util
import os
import sys

import librosa
import numpy as np

from .models import BranchScoreBreakdown, JobOptions, NoteEvent, TechniqueHint, TranscriptionProfile

GUITAR_MIN_MIDI = 40
GUITAR_MAX_MIDI = 88


@dataclass
class AnalysisFeatures:
    sample_rate: int
    onset_times: np.ndarray
    onset_strength: np.ndarray
    rms_times: np.ndarray
    rms_values: np.ndarray
    pyin_times: np.ndarray | None = None
    pyin_midi: np.ndarray | None = None


@dataclass
class BranchTranscription:
    branch_name: str
    raw_note_events: list[NoteEvent]
    note_events: list[NoteEvent]
    average_confidence: float
    warnings: list[str]
    cleanup_warnings: list[str]
    score: BranchScoreBreakdown
    used_fallback: bool = False


def transcribe_with_basic_pitch(input_path: str, branch_name: str) -> list[NoteEvent]:
    basic_pitch, inference = import_basic_pitch_with_preferred_backend()

    model_path = pick_basic_pitch_model_path(
        basic_pitch.FilenameSuffix,
        basic_pitch.build_icassp_2022_model_path,
    )
    _, _, raw_note_events = inference.predict(input_path, model_or_model_path=model_path)
    events: list[NoteEvent] = []
    for item in raw_note_events:
        if len(item) >= 5:
            onset, offset, midi_pitch, confidence, pitch_bend = item[:5]
        elif len(item) == 4:
            onset, offset, midi_pitch, confidence = item
            pitch_bend = 0.0
        else:
            continue

        pitch_bend_cents = normalize_pitch_bend(pitch_bend)
        technique = TechniqueHint.bend if abs(pitch_bend_cents) >= 25 else TechniqueHint.picked
        events.append(
            NoteEvent(
                onset_sec=float(onset),
                offset_sec=float(offset),
                midi_pitch=int(round(midi_pitch)),
                confidence=float(confidence),
                pitch_bend_cents=pitch_bend_cents,
                technique_hint=technique,
                source_branch=branch_name,
            )
        )
    return merge_adjacent_notes(events)


def transcribe_with_fallback(input_path: str, branch_name: str) -> list[NoteEvent]:
    audio, sample_rate = librosa.load(input_path, sr=16000, mono=True)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        sr=sample_rate,
        fmin=librosa.note_to_hz("E2"),
        fmax=librosa.note_to_hz("E6"),
        frame_length=2048,
    )
    times = librosa.times_like(f0, sr=sample_rate)

    events: list[NoteEvent] = []
    active_start = None
    active_pitch = None
    active_conf = []
    last_time = 0.0

    for time_value, pitch_hz, voiced, confidence in zip(times, f0, voiced_flag, voiced_probs):
        last_time = float(time_value)
        if voiced and pitch_hz is not None:
            midi_pitch = int(round(librosa.hz_to_midi(float(pitch_hz))))
            if active_pitch is None:
                active_start = time_value
                active_pitch = midi_pitch
                active_conf = [float(confidence or 0.0)]
            elif abs(active_pitch - midi_pitch) <= 1:
                active_conf.append(float(confidence or 0.0))
                active_pitch = int(round((active_pitch + midi_pitch) / 2))
            else:
                events.append(
                    NoteEvent(
                        onset_sec=float(active_start),
                        offset_sec=float(time_value),
                        midi_pitch=active_pitch,
                        confidence=float(np.mean(active_conf)) if active_conf else 0.0,
                        source_branch=branch_name,
                    )
                )
                active_start = time_value
                active_pitch = midi_pitch
                active_conf = [float(confidence or 0.0)]
        elif active_pitch is not None:
            events.append(
                NoteEvent(
                    onset_sec=float(active_start),
                    offset_sec=float(time_value),
                    midi_pitch=active_pitch,
                    confidence=float(np.mean(active_conf)) if active_conf else 0.0,
                    source_branch=branch_name,
                )
            )
            active_start = None
            active_pitch = None
            active_conf = []

    if active_pitch is not None and active_start is not None:
        events.append(
            NoteEvent(
                onset_sec=float(active_start),
                offset_sec=last_time,
                midi_pitch=active_pitch,
                confidence=float(np.mean(active_conf)) if active_conf else 0.0,
                source_branch=branch_name,
            )
        )

    return merge_adjacent_notes(events)


def transcribe_audio(input_path: str, branch_name: str, options: JobOptions | None = None) -> BranchTranscription:
    options = options or JobOptions()
    warnings: list[str] = []
    used_fallback = False
    disable_basic_pitch = os.environ.get("MUSIC_TAB_DISABLE_BASIC_PITCH") == "1"
    try:
        if disable_basic_pitch:
            raise RuntimeError("Basic Pitch disabled by environment.")
        raw_events = transcribe_with_basic_pitch(input_path, branch_name)
    except Exception as exc:  # pragma: no cover - Basic Pitch optional path
        warnings.append(f"Basic Pitch unavailable; used librosa fallback ({exc}).")
        raw_events = transcribe_with_fallback(input_path, branch_name)
        used_fallback = True

    refined_events, refinement_warnings = refine_note_events_with_audio(raw_events, input_path, options)
    cleaned_events, cleanup_warnings = cleanup_note_events(
        refined_events,
        lead_guitar_mode=options.lead_guitar_mode,
    )
    warnings.extend(refinement_warnings)
    average_confidence = (
        sum(event.confidence for event in cleaned_events) / len(cleaned_events) if cleaned_events else 0.0
    )
    score = score_branch(input_path, cleaned_events, average_confidence, options)
    return BranchTranscription(
        branch_name=branch_name,
        raw_note_events=raw_events,
        note_events=cleaned_events,
        average_confidence=average_confidence,
        warnings=warnings,
        cleanup_warnings=cleanup_warnings,
        score=score,
        used_fallback=used_fallback,
    )


def choose_best_branch(branches: list[BranchTranscription]) -> BranchTranscription:
    if not branches:
        return BranchTranscription(
            branch_name="original",
            raw_note_events=[],
            note_events=[],
            average_confidence=0.0,
            warnings=["No transcription branches were produced."],
            cleanup_warnings=[],
            score=BranchScoreBreakdown(),
        )
    return max(branches, key=lambda branch: branch.score.overall)


def fuse_branch_events(
    primary: BranchTranscription,
    alternate: BranchTranscription,
    options: JobOptions | None = None,
) -> tuple[list[NoteEvent], bool]:
    options = options or JobOptions()
    if options.transcription_profile != TranscriptionProfile.accurate:
        return list(primary.note_events), False
    if not primary.note_events or not alternate.note_events:
        return list(primary.note_events), False

    fused = list(primary.note_events)
    replaced = False
    for alt_event in alternate.note_events:
        if local_polyphony(alternate.note_events, alt_event.onset_sec) > 2:
            continue

        match_index = find_matching_event(fused, alt_event)
        if match_index is None:
            if alt_event.confidence >= 0.72:
                fused.append(alt_event)
                replaced = True
            continue

        current = fused[match_index]
        if alt_event.confidence >= current.confidence + 0.12 and onset_delta(current, alt_event) <= 0.06:
            fused[match_index] = alt_event
            replaced = True

    cleaned_events, _ = cleanup_note_events(
        fused,
        lead_guitar_mode=options.lead_guitar_mode,
    )
    return cleaned_events, replaced


def should_attempt_separation(input_path: str, options: JobOptions) -> tuple[bool, str]:
    if options.separation_mode.value == "off":
        return False, "Separation disabled."
    if options.separation_mode.value == "demucs":
        return True, "Separation explicitly requested."

    features = load_analysis_features(input_path, include_pyin=False)
    onset_frames = int(np.sum(features.onset_strength > 0.18))
    duration_sec = max(float(features.onset_times[-1]) if len(features.onset_times) else 0.0, 1.0)
    onset_density = onset_frames / duration_sec
    rms_spread = float(np.std(features.rms_values)) if len(features.rms_values) else 0.0
    crowded_mix = onset_density > 2.8 or rms_spread > 0.22
    accurate_mode = options.transcription_profile == TranscriptionProfile.accurate
    return crowded_mix and accurate_mode, (
        "Auto separation enabled due to crowded mix."
        if crowded_mix and accurate_mode
        else "Auto separation skipped; mix appears sparse enough."
    )


def pick_basic_pitch_model_path(filename_suffix, build_model_path):
    preferred_backend = os.environ.get("MUSIC_TAB_BASIC_PITCH_BACKEND", "onnx").lower()
    backend_order = [preferred_backend, "onnx", "tflite", "tf"]
    seen: set[str] = set()

    availability = {
        "onnx": importlib.util.find_spec("onnxruntime") is not None,
        "tflite": importlib.util.find_spec("tflite_runtime") is not None
        or importlib.util.find_spec("tensorflow") is not None,
        "tf": importlib.util.find_spec("tensorflow") is not None,
    }
    suffix_map = {
        "onnx": filename_suffix.onnx,
        "tflite": filename_suffix.tflite,
        "tf": filename_suffix.tf,
    }

    for backend in backend_order:
        if backend in seen:
            continue
        seen.add(backend)
        if not availability.get(backend, False):
            continue
        candidate_path = build_model_path(suffix_map[backend])
        if candidate_path.exists():
            return candidate_path

    raise RuntimeError("No usable Basic Pitch model backend is available.")


def import_basic_pitch_with_preferred_backend():
    blocked_roots = blocked_backend_roots()
    for module_name in list(sys.modules):
        if module_name == "basic_pitch" or module_name.startswith("basic_pitch."):
            del sys.modules[module_name]

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root_name = name.split(".")[0]
        if root_name in blocked_roots:
            raise ImportError(f"{root_name} intentionally disabled for Basic Pitch backend selection.")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    try:
        import importlib

        basic_pitch = importlib.import_module("basic_pitch")
        inference = importlib.import_module("basic_pitch.inference")
        return basic_pitch, inference
    finally:
        builtins.__import__ = original_import


def blocked_backend_roots() -> set[str]:
    preferred_backend = os.environ.get("MUSIC_TAB_BASIC_PITCH_BACKEND", "onnx").lower()
    if preferred_backend == "onnx":
        return {"tensorflow", "coremltools", "tflite_runtime"}
    if preferred_backend == "tflite":
        return {"tensorflow", "coremltools", "onnxruntime"}
    if preferred_backend == "tf":
        return {"coremltools", "onnxruntime", "tflite_runtime"}
    return set()


def normalize_pitch_bend(raw_pitch_bend) -> float:
    if raw_pitch_bend is None:
        return 0.0
    if isinstance(raw_pitch_bend, (list, tuple)):
        if not raw_pitch_bend:
            return 0.0
        return float(np.mean(raw_pitch_bend)) * 100.0
    return float(raw_pitch_bend)


def refine_note_events_with_audio(
    events: list[NoteEvent],
    input_path: str,
    options: JobOptions,
) -> tuple[list[NoteEvent], list[str]]:
    if not events:
        return [], []

    include_pyin = options.transcription_profile == TranscriptionProfile.accurate
    features = load_analysis_features(input_path, include_pyin=include_pyin)
    warnings: list[str] = []
    refined: list[NoteEvent] = []
    ordered = sorted(events, key=lambda item: (item.onset_sec, item.offset_sec, item.midi_pitch))

    for index, event in enumerate(ordered):
        onset_support = signal_peak(features.onset_times, features.onset_strength, event.onset_sec, 0.06)
        duration = event.offset_sec - event.onset_sec
        if onset_support < 0.08 and event.confidence < 0.48 and duration < 0.12:
            continue

        snapped_onset = snap_to_onset_peak(event.onset_sec, features.onset_times, features.onset_strength)
        next_onset = ordered[index + 1].onset_sec if index + 1 < len(ordered) else None
        snapped_offset = snap_offset_to_decay(event, features, snapped_onset, next_onset)
        if snapped_offset - snapped_onset < 0.04:
            snapped_offset = snapped_onset + 0.04

        refined.append(
            event.model_copy(
                update={
                    "onset_sec": snapped_onset,
                    "offset_sec": snapped_offset,
                }
            )
        )

    if include_pyin and features.pyin_midi is not None and features.pyin_times is not None:
        refined = refine_monophonic_pitches(refined, features)
        if len(refined) != len(events):
            warnings.append("Low-support note attacks were removed during onset refinement.")

    return refined, warnings


def cleanup_note_events(
    events: list[NoteEvent],
    lead_guitar_mode: bool = True,
) -> tuple[list[NoteEvent], list[str]]:
    warnings: list[str] = []
    filtered: list[NoteEvent] = []
    invalid_range = 0
    short_count = 0

    for event in sorted(events, key=lambda item: (item.onset_sec, item.midi_pitch, item.offset_sec)):
        if event.offset_sec - event.onset_sec < 0.04:
            short_count += 1
            continue
        if event.midi_pitch < GUITAR_MIN_MIDI or event.midi_pitch > GUITAR_MAX_MIDI:
            invalid_range += 1
            continue
        filtered.append(event)

    if short_count:
        warnings.append(f"Removed {short_count} ultra-short note fragments.")
    if invalid_range:
        warnings.append(f"Removed {invalid_range} out-of-range guitar notes.")

    merged: list[NoteEvent] = []
    duplicate_merges = 0
    for event in filtered:
        if merged:
            previous = merged[-1]
            same_pitch = previous.midi_pitch == event.midi_pitch
            close_gap = event.onset_sec - previous.offset_sec <= 0.06
            overlapping_duplicate = same_pitch and event.onset_sec <= previous.offset_sec + 0.02
            if same_pitch and (close_gap or overlapping_duplicate):
                duplicate_merges += 1
                merged[-1] = previous.model_copy(
                    update={
                        "offset_sec": max(previous.offset_sec, event.offset_sec),
                        "confidence": max(previous.confidence, event.confidence),
                        "pitch_bend_cents": previous.pitch_bend_cents
                        if abs(previous.pitch_bend_cents) >= abs(event.pitch_bend_cents)
                        else event.pitch_bend_cents,
                    }
                )
                continue
        merged.append(event)

    if duplicate_merges:
        warnings.append(f"Merged {duplicate_merges} duplicate or over-segmented note pairs.")

    pruned: list[NoteEvent] = []
    octave_prunes = 0
    max_polyphony = 4 if lead_guitar_mode else 6
    simultaneous_prunes = 0
    for group in group_by_onset(merged, 0.05):
        kept = list(group)
        for event in list(kept):
            for other in kept:
                if event is other:
                    continue
                interval = abs(event.midi_pitch - other.midi_pitch)
                if interval in {11, 12, 13} and event.confidence + 0.1 < other.confidence:
                    if event in kept:
                        kept.remove(event)
                        octave_prunes += 1
                    break
        if len(kept) > max_polyphony:
            kept = sorted(kept, key=lambda item: (item.confidence, item.offset_sec - item.onset_sec), reverse=True)[
                :max_polyphony
            ]
            simultaneous_prunes += max(0, len(group) - len(kept))
        pruned.extend(sorted(kept, key=lambda item: (item.onset_sec, item.midi_pitch)))

    if octave_prunes:
        warnings.append(f"Suppressed {octave_prunes} likely octave-ghost notes.")
    if simultaneous_prunes:
        warnings.append(f"Reduced {simultaneous_prunes} excessive simultaneous note attacks.")
    final_events: list[NoteEvent] = []
    for event in sorted(pruned, key=lambda item: (item.onset_sec, item.midi_pitch)):
        if final_events:
            previous = final_events[-1]
            if previous.midi_pitch == event.midi_pitch and event.onset_sec - previous.offset_sec <= 0.06:
                final_events[-1] = previous.model_copy(
                    update={
                        "offset_sec": max(previous.offset_sec, event.offset_sec),
                        "confidence": max(previous.confidence, event.confidence),
                    }
                )
                continue
        final_events.append(event)
    return final_events, warnings


def merge_adjacent_notes(events: list[NoteEvent]) -> list[NoteEvent]:
    return cleanup_note_events(events)[0]


def score_branch(
    input_path: str,
    events: list[NoteEvent],
    average_confidence: float,
    options: JobOptions,
) -> BranchScoreBreakdown:
    if not events:
        return BranchScoreBreakdown()

    features = load_analysis_features(input_path, include_pyin=False)
    onset_alignment = float(
        np.mean(
            [
                signal_peak(features.onset_times, features.onset_strength, event.onset_sec, 0.05)
                for event in events
            ]
        )
    )
    playable_ratio = playable_note_ratio(events, options)
    duplicate_penalty = duplicate_pitch_ratio(events)
    octave_penalty = octave_overlap_ratio(events)
    density_score = density_sanity_score(events, lead_guitar_mode=options.lead_guitar_mode)
    overall = (
        average_confidence * 0.32
        + onset_alignment * 0.24
        + playable_ratio * 0.2
        + density_score * 0.14
        - duplicate_penalty * 0.12
        - octave_penalty * 0.08
    )
    return BranchScoreBreakdown(
        average_confidence=average_confidence,
        onset_alignment=onset_alignment,
        playable_ratio=playable_ratio,
        density_score=density_score,
        duplicate_penalty=duplicate_penalty,
        octave_penalty=octave_penalty,
        overall=overall,
    )


@lru_cache(maxsize=16)
def load_analysis_features(input_path: str, include_pyin: bool) -> AnalysisFeatures:
    audio, sample_rate = librosa.load(input_path, sr=22050, mono=True)
    if not len(audio):
        return AnalysisFeatures(
            sample_rate=sample_rate,
            onset_times=np.array([]),
            onset_strength=np.array([]),
            rms_times=np.array([]),
            rms_values=np.array([]),
        )

    harmonic, _ = librosa.effects.hpss(audio)
    onset_strength = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    onset_strength = normalize_signal(onset_strength)
    onset_times = librosa.times_like(onset_strength, sr=sample_rate)

    rms_values = librosa.feature.rms(y=harmonic, frame_length=2048, hop_length=512)[0]
    rms_values = normalize_signal(rms_values)
    rms_times = librosa.times_like(rms_values, sr=sample_rate, hop_length=512)

    pyin_times = None
    pyin_midi = None
    if include_pyin:
        f0, voiced_flag, _ = librosa.pyin(
            harmonic,
            sr=sample_rate,
            fmin=librosa.note_to_hz("E2"),
            fmax=librosa.note_to_hz("E6"),
            frame_length=2048,
        )
        pyin_times = librosa.times_like(f0, sr=sample_rate)
        pyin_midi = np.where(voiced_flag, librosa.hz_to_midi(f0), np.nan)

    return AnalysisFeatures(
        sample_rate=sample_rate,
        onset_times=onset_times,
        onset_strength=onset_strength,
        rms_times=rms_times,
        rms_values=rms_values,
        pyin_times=pyin_times,
        pyin_midi=pyin_midi,
    )


def normalize_signal(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    scale = float(np.percentile(values, 95))
    if scale <= 1e-6:
        scale = float(np.max(values)) or 1.0
    return np.clip(values / scale, 0.0, 1.5)


def signal_peak(times: np.ndarray, values: np.ndarray, center: float, radius: float) -> float:
    if values.size == 0 or times.size == 0:
        return 0.0
    mask = (times >= center - radius) & (times <= center + radius)
    if np.any(mask):
        return float(np.max(values[mask]))
    index = int(np.argmin(np.abs(times - center)))
    return float(values[index])


def snap_to_onset_peak(onset_sec: float, onset_times: np.ndarray, onset_strength: np.ndarray) -> float:
    if onset_times.size == 0 or onset_strength.size == 0:
        return onset_sec
    mask = (onset_times >= onset_sec - 0.04) & (onset_times <= onset_sec + 0.06)
    if not np.any(mask):
        return onset_sec
    local_times = onset_times[mask]
    local_values = onset_strength[mask]
    peak_index = int(np.argmax(local_values))
    if float(local_values[peak_index]) < 0.1:
        return onset_sec
    return float(local_times[peak_index])


def snap_offset_to_decay(
    event: NoteEvent,
    features: AnalysisFeatures,
    onset_sec: float,
    next_onset_sec: float | None,
) -> float:
    if features.rms_times.size == 0 or features.rms_values.size == 0:
        return event.offset_sec

    end_limit = event.offset_sec + 0.18
    if next_onset_sec is not None:
        end_limit = min(end_limit, next_onset_sec - 0.01)
    mask = (features.rms_times >= onset_sec) & (features.rms_times <= end_limit)
    if not np.any(mask):
        return max(event.offset_sec, onset_sec + 0.04)

    local_times = features.rms_times[mask]
    local_values = features.rms_values[mask]
    local_peak = float(np.max(local_values)) or 0.1
    threshold = max(0.08, local_peak * 0.22)
    active = np.where(local_values >= threshold)[0]
    if active.size == 0:
        return max(event.offset_sec, onset_sec + 0.04)
    return max(float(local_times[active[-1]]) + 0.03, onset_sec + 0.04)


def refine_monophonic_pitches(events: list[NoteEvent], features: AnalysisFeatures) -> list[NoteEvent]:
    if features.pyin_times is None or features.pyin_midi is None:
        return events

    refined: list[NoteEvent] = []
    for event in events:
        if local_polyphony(events, (event.onset_sec + event.offset_sec) / 2) > 1:
            refined.append(event)
            continue
        mask = (features.pyin_times >= event.onset_sec) & (features.pyin_times <= event.offset_sec)
        samples = features.pyin_midi[mask]
        samples = samples[np.isfinite(samples)]
        if samples.size < 3:
            refined.append(event)
            continue
        refined_pitch = int(round(float(np.nanmedian(samples))))
        delta = abs(refined_pitch - event.midi_pitch)
        if delta <= 1 or (delta in {11, 12, 13} and event.confidence < 0.75):
            refined.append(event.model_copy(update={"midi_pitch": refined_pitch}))
            continue
        refined.append(event)
    return refined


def group_by_onset(events: list[NoteEvent], tolerance: float) -> list[list[NoteEvent]]:
    groups: list[list[NoteEvent]] = []
    current: list[NoteEvent] = []
    current_start: float | None = None
    for event in events:
        if current_start is None or event.onset_sec - current_start <= tolerance:
            current.append(event)
            if current_start is None:
                current_start = event.onset_sec
            continue
        groups.append(current)
        current = [event]
        current_start = event.onset_sec
    if current:
        groups.append(current)
    return groups


def playable_note_ratio(events: list[NoteEvent], options: JobOptions) -> float:
    from .tab_generator import generate_tab_events

    if not events:
        return 0.0
    tab_events = generate_tab_events(events, options)
    return len(tab_events) / len(events)


def duplicate_pitch_ratio(events: list[NoteEvent]) -> float:
    if not events:
        return 0.0
    duplicates = 0
    ordered = sorted(events, key=lambda item: (item.onset_sec, item.midi_pitch))
    for previous, current in zip(ordered, ordered[1:]):
        same_pitch = previous.midi_pitch == current.midi_pitch
        close = current.onset_sec <= previous.offset_sec + 0.03
        if same_pitch and close:
            duplicates += 1
    return duplicates / len(events)


def octave_overlap_ratio(events: list[NoteEvent]) -> float:
    if not events:
        return 0.0
    octave_pairs = 0
    for group in group_by_onset(sorted(events, key=lambda item: item.onset_sec), 0.05):
        pitches = [event.midi_pitch for event in group]
        for index, pitch in enumerate(pitches):
            if any(abs(pitch - other) in {11, 12, 13} for other in pitches[index + 1 :]):
                octave_pairs += 1
                break
    return octave_pairs / len(events)


def density_sanity_score(events: list[NoteEvent], lead_guitar_mode: bool) -> float:
    if not events:
        return 0.0
    duration = max(events[-1].offset_sec - events[0].onset_sec, 0.5)
    notes_per_sec = len(events) / duration
    target_density = 6.5 if lead_guitar_mode else 8.0
    density_penalty = max(0.0, notes_per_sec - target_density) / target_density
    peak_polyphony = max((len(group) for group in group_by_onset(events, 0.05)), default=1)
    polyphony_penalty = max(0.0, peak_polyphony - (4 if lead_guitar_mode else 6)) / 4
    return max(0.0, 1.0 - density_penalty - polyphony_penalty)


def local_polyphony(events: list[NoteEvent], time_point: float) -> int:
    return sum(1 for event in events if event.onset_sec <= time_point < event.offset_sec)


def find_matching_event(events: list[NoteEvent], candidate: NoteEvent) -> int | None:
    for index, event in enumerate(events):
        if event.midi_pitch != candidate.midi_pitch:
            continue
        if onset_delta(event, candidate) <= 0.08:
            return index
    return None


def onset_delta(left: NoteEvent, right: NoteEvent) -> float:
    return abs(left.onset_sec - right.onset_sec)
