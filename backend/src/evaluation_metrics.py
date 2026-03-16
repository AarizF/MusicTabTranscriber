from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .models import NoteEvent, TabEvent


@dataclass(frozen=True)
class MatchResult:
    reference_index: int
    predicted_index: int


def compute_note_metrics(
    reference_events: list[NoteEvent],
    predicted_events: list[NoteEvent],
) -> dict[str, float]:
    metrics = {
        "onset_precision": 0.0,
        "onset_recall": 0.0,
        "onset_f1": 0.0,
        "note_precision": 0.0,
        "note_recall": 0.0,
        "note_f1": 0.0,
        "pitch_accuracy": 0.0,
    }
    if not reference_events and not predicted_events:
        return {key: 1.0 for key in metrics}
    if not reference_events or not predicted_events:
        return metrics

    try:
        import mir_eval
    except Exception as exc:  # pragma: no cover - dependency should exist in app env
        raise RuntimeError("mir_eval is required for note-level evaluation.") from exc

    reference_intervals, reference_pitches = events_to_mir_eval_inputs(reference_events)
    predicted_intervals, predicted_pitches = events_to_mir_eval_inputs(predicted_events)

    onset_precision, onset_recall, onset_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        reference_intervals,
        reference_pitches,
        predicted_intervals,
        predicted_pitches,
        onset_tolerance=0.05,
        offset_ratio=None,
    )
    note_precision, note_recall, note_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        reference_intervals,
        reference_pitches,
        predicted_intervals,
        predicted_pitches,
        onset_tolerance=0.05,
        offset_ratio=0.2,
    )
    onset_matches = greedy_match(reference_events, predicted_events, require_offset=False)
    pitch_accuracy = 0.0
    if onset_matches:
        pitch_accuracy = sum(
            1
            for match in onset_matches
            if reference_events[match.reference_index].midi_pitch == predicted_events[match.predicted_index].midi_pitch
        ) / len(onset_matches)

    metrics.update(
        {
            "onset_precision": float(onset_precision),
            "onset_recall": float(onset_recall),
            "onset_f1": float(onset_f1),
            "note_precision": float(note_precision),
            "note_recall": float(note_recall),
            "note_f1": float(note_f1),
            "pitch_accuracy": float(pitch_accuracy),
        }
    )
    return metrics


def compute_tab_metrics(
    reference_events: list[TabEvent],
    predicted_events: list[TabEvent],
) -> dict[str, float]:
    metrics = {
        "string_accuracy": 0.0,
        "fret_accuracy": 0.0,
        "invalid_fingering_rate": invalid_fingering_rate(predicted_events),
        "voice_density_peak": voice_density_peak(predicted_events),
    }
    if not reference_events or not predicted_events:
        return metrics

    matches = greedy_match(reference_events, predicted_events, require_offset=False)
    if not matches:
        return metrics

    string_matches = sum(
        1
        for match in matches
        if reference_events[match.reference_index].string == predicted_events[match.predicted_index].string
    )
    fret_matches = sum(
        1
        for match in matches
        if reference_events[match.reference_index].fret == predicted_events[match.predicted_index].fret
    )
    metrics["string_accuracy"] = string_matches / len(matches)
    metrics["fret_accuracy"] = fret_matches / len(matches)
    return metrics


def events_to_mir_eval_inputs(events: list[NoteEvent]) -> tuple[np.ndarray, np.ndarray]:
    intervals = np.array([[event.onset_sec, max(event.offset_sec, event.onset_sec + 0.01)] for event in events])
    pitches = librosa.midi_to_hz(np.array([event.midi_pitch for event in events], dtype=float))
    return intervals, pitches


def greedy_match(
    reference_events: list[NoteEvent],
    predicted_events: list[NoteEvent],
    require_offset: bool,
) -> list[MatchResult]:
    remaining = set(range(len(predicted_events)))
    matches: list[MatchResult] = []
    for reference_index, reference in enumerate(reference_events):
        best_index = None
        best_distance = None
        for predicted_index in remaining:
            predicted = predicted_events[predicted_index]
            if abs(reference.midi_pitch - predicted.midi_pitch) > 0:
                continue
            onset_distance = abs(reference.onset_sec - predicted.onset_sec)
            if onset_distance > 0.05:
                continue
            if require_offset:
                ref_duration = max(reference.offset_sec - reference.onset_sec, 0.05)
                offset_delta = abs(reference.offset_sec - predicted.offset_sec)
                if offset_delta > max(0.05, ref_duration * 0.2):
                    continue
                distance = onset_distance + offset_delta
            else:
                distance = onset_distance
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = predicted_index
        if best_index is None:
            continue
        remaining.remove(best_index)
        matches.append(MatchResult(reference_index=reference_index, predicted_index=best_index))
    return matches


def invalid_fingering_rate(tab_events: list[TabEvent]) -> float:
    if not tab_events:
        return 0.0
    invalid = 0
    for event in tab_events:
        if not (1 <= event.string <= 6 and event.fret >= 0):
            invalid += 1
    return invalid / len(tab_events)


def voice_density_peak(tab_events: list[TabEvent]) -> float:
    if not tab_events:
        return 0.0
    peak = 0
    for event in tab_events:
        density = sum(1 for other in tab_events if other.onset_sec <= event.onset_sec < other.offset_sec)
        peak = max(peak, density)
    return float(peak)

