from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import librosa
import numpy as np

from .models import BeatGrid, JobOptions, QuantizedNote, ScoreEvent, ScoreMeasure, ScorePart, TabEvent


DIVISIONS_PER_QUARTER = 4
SLOTS_PER_MEASURE = 16
DURATION_PIECES = [
    (16, "whole", 0),
    (12, "half", 1),
    (8, "half", 0),
    (6, "quarter", 1),
    (4, "quarter", 0),
    (3, "eighth", 1),
    (2, "eighth", 0),
    (1, "16th", 0),
]


@dataclass
class _Piece:
    measure_index: int
    start_slot: int
    duration_slots: int
    note_type: str
    dots: int
    note: QuantizedNote


def build_score_part(
    title: str,
    tab_events: list[TabEvent],
    audio_path: str,
    duration_sec: float,
    options: JobOptions,
) -> tuple[ScorePart | None, list[str]]:
    warnings: list[str] = []
    beat_grid = infer_beat_grid(audio_path, duration_sec, options)
    if beat_grid.confidence < 0.3:
        warnings.append("Beat grid confidence was too low for engraved output.")
        return None, warnings

    measure_count = max(1, int(ceil(duration_sec / max(beat_grid.beat_duration_sec * beat_grid.beats_per_measure, 0.001))))
    note_pieces, quantization_error_sec = quantize_tab_events(tab_events, beat_grid, measure_count)
    measures, measure_warnings = build_measures(note_pieces, measure_count)
    warnings.extend(measure_warnings)

    score_part = ScorePart(
        title=title,
        time_signature=f"{beat_grid.beats_per_measure}/{beat_grid.beat_unit}",
        divisions=DIVISIONS_PER_QUARTER,
        beat_grid=beat_grid,
        measures=measures,
        quantization_error_sec=quantization_error_sec,
        measure_fill_valid=all(measure_voices_fill(measure) for measure in measures),
    )

    if not score_part.measure_fill_valid:
        warnings.append("Measure fill validation failed; engraved output has been skipped.")
        return None, warnings

    return score_part, warnings


def infer_beat_grid(audio_path: str, duration_sec: float, options: JobOptions) -> BeatGrid:
    beats_per_measure = 4
    beat_unit = 4
    time_signature = options.time_signature_override or options.time_signature_hint or "4/4"
    if time_signature != "4/4":
        time_signature = "4/4"

    effective_tempo = options.tempo_override_bpm or options.tempo_hint_bpm
    if effective_tempo:
        bpm = float(effective_tempo)
        confidence = 1.0
    else:
        audio, sample_rate = librosa.load(audio_path, sr=22050, mono=True)
        onset_envelope = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        detected_tempo, tracked_beats = librosa.beat.beat_track(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            units="time",
        )
        bpm = float(np.atleast_1d(detected_tempo)[0]) if np.size(detected_tempo) else 120.0
        if bpm <= 0:
            bpm = 120.0

        tracked_beats = np.atleast_1d(tracked_beats)
        if tracked_beats.size >= 4:
            intervals = np.diff(tracked_beats.astype(float))
            regularity = float(np.std(intervals) / max(np.mean(intervals), 1e-6))
            confidence = max(0.0, min(1.0, 1.0 - regularity * 4.0))
        else:
            confidence = 0.45 if duration_sec <= 3.0 else 0.25

    beat_duration_sec = 60.0 / bpm
    subdivision_duration_sec = beat_duration_sec / DIVISIONS_PER_QUARTER
    beat_times = list(np.arange(0.0, duration_sec + beat_duration_sec, beat_duration_sec))
    return BeatGrid(
        bpm=bpm,
        beat_times=beat_times,
        beat_duration_sec=beat_duration_sec,
        subdivision_duration_sec=subdivision_duration_sec,
        beats_per_measure=beats_per_measure,
        beat_unit=beat_unit,
        confidence=confidence,
    )


def quantize_tab_events(
    tab_events: list[TabEvent],
    beat_grid: BeatGrid,
    measure_count: int,
) -> tuple[list[_Piece], float]:
    pieces: list[_Piece] = []
    total_error = 0.0
    total_points = 0
    for event in tab_events:
        onset_slot_abs = max(0, int(round(event.onset_sec / beat_grid.subdivision_duration_sec)))
        offset_slot_abs = max(
            onset_slot_abs + 1,
            int(ceil(event.offset_sec / beat_grid.subdivision_duration_sec)),
        )
        quantized_onset_sec = onset_slot_abs * beat_grid.subdivision_duration_sec
        quantized_offset_sec = offset_slot_abs * beat_grid.subdivision_duration_sec
        total_error += abs(event.onset_sec - quantized_onset_sec) + abs(event.offset_sec - quantized_offset_sec)
        total_points += 2

        current_slot_abs = onset_slot_abs
        first_piece = True
        while current_slot_abs < offset_slot_abs:
            measure_index = current_slot_abs // SLOTS_PER_MEASURE
            if measure_index >= measure_count:
                break
            measure_limit_abs = (measure_index + 1) * SLOTS_PER_MEASURE
            span_in_measure = min(offset_slot_abs, measure_limit_abs) - current_slot_abs
            local_offset_abs = current_slot_abs + span_in_measure
            local_slot_abs = current_slot_abs
            for piece_slots, note_type, dots in decompose_duration_slots(span_in_measure):
                continuation_from_previous = not first_piece or local_slot_abs > onset_slot_abs
                continues_after = local_slot_abs + piece_slots < offset_slot_abs
                pieces.append(
                    _Piece(
                        measure_index=measure_index,
                        start_slot=local_slot_abs % SLOTS_PER_MEASURE,
                        duration_slots=piece_slots,
                        note_type=note_type,
                        dots=dots,
                        note=QuantizedNote(
                            onset_slot=local_slot_abs % SLOTS_PER_MEASURE,
                            duration_slots=piece_slots,
                            midi_pitch=event.midi_pitch,
                            string=event.string,
                            fret=event.fret,
                            confidence=event.confidence,
                            pitch_bend_cents=event.pitch_bend_cents,
                            technique_hint=event.technique_hint,
                            tie_start=continues_after,
                            tie_stop=continuation_from_previous,
                        ),
                    )
                )
                local_slot_abs += piece_slots
            current_slot_abs = local_offset_abs
            first_piece = False
    average_error = total_error / total_points if total_points else 0.0
    return pieces, average_error


def build_measures(note_pieces: list[_Piece], measure_count: int) -> tuple[list[ScoreMeasure], list[str]]:
    warnings: list[str] = []
    highly_polyphonic_measures: list[int] = []
    measures: list[ScoreMeasure] = []
    by_measure: dict[int, list[_Piece]] = {}
    for piece in note_pieces:
        by_measure.setdefault(piece.measure_index, []).append(piece)

    for measure_index in range(measure_count):
        grouped: dict[tuple[int, int, str, int], list[QuantizedNote]] = {}
        for piece in sorted(
            by_measure.get(measure_index, []),
            key=lambda item: (item.start_slot, item.duration_slots, item.note.midi_pitch),
        ):
            key = (piece.start_slot, piece.duration_slots, piece.note_type, piece.dots)
            grouped.setdefault(key, []).append(piece.note)

        note_events = []
        for (start_slot, duration_slots, note_type, dots), notes in grouped.items():
            sorted_notes = sorted(notes, key=lambda item: item.midi_pitch)
            for index, note in enumerate(sorted_notes):
                note.is_chord_tone = index > 0
            note_events.append(
                ScoreEvent(
                    start_slot=start_slot,
                    duration_slots=duration_slots,
                    note_type=note_type,
                    dots=dots,
                    notes=sorted_notes,
                )
            )
        note_events.sort(key=lambda event: (event.start_slot, event.duration_slots))
        voiced_events, voice_count = assign_voices(note_events)
        if voice_count > 4:
            highly_polyphonic_measures.append(measure_index + 1)

        measure_events: list[ScoreEvent] = []
        for voice_index in range(1, voice_count + 1):
            voice_events = [event for event in voiced_events if event.voice == voice_index]
            current_slot = 0
            for event in voice_events:
                if event.start_slot > current_slot:
                    measure_events.extend(create_rest_events(current_slot, event.start_slot - current_slot, voice_index))
                measure_events.append(event)
                current_slot = max(current_slot, event.start_slot + event.duration_slots)
            if current_slot < SLOTS_PER_MEASURE:
                measure_events.extend(create_rest_events(current_slot, SLOTS_PER_MEASURE - current_slot, voice_index))

        measures.append(
            ScoreMeasure(
                number=measure_index + 1,
                total_slots=SLOTS_PER_MEASURE,
                voice_count=voice_count,
                events=measure_events,
            )
        )

    if highly_polyphonic_measures:
        warnings.append(
            "Detected highly polyphonic overlap in "
            f"{len(highly_polyphonic_measures)} measures ({format_measure_ranges(highly_polyphonic_measures)}); "
            "engraving may be visually dense."
        )

    return measures, warnings


def assign_voices(note_events: list[ScoreEvent]) -> tuple[list[ScoreEvent], int]:
    voice_end_slots: list[int] = []
    assigned_events: list[ScoreEvent] = []

    for event in note_events:
        assigned_voice = None
        for voice_index, end_slot in enumerate(voice_end_slots, start=1):
            if end_slot <= event.start_slot:
                assigned_voice = voice_index
                break

        if assigned_voice is None:
            voice_end_slots.append(0)
            assigned_voice = len(voice_end_slots)

        voice_end_slots[assigned_voice - 1] = max(
            voice_end_slots[assigned_voice - 1],
            event.start_slot + event.duration_slots,
        )
        assigned_events.append(event.model_copy(update={"voice": assigned_voice}))

    used_voice_count = max((event.voice for event in assigned_events), default=1)
    assigned_events.sort(key=lambda event: (event.voice, event.start_slot, event.duration_slots))
    return assigned_events, used_voice_count


def create_rest_events(start_slot: int, duration_slots: int, voice: int) -> list[ScoreEvent]:
    events: list[ScoreEvent] = []
    cursor = start_slot
    for piece_slots, note_type, dots in decompose_duration_slots(duration_slots):
        events.append(
            ScoreEvent(
                voice=voice,
                start_slot=cursor,
                duration_slots=piece_slots,
                note_type=note_type,
                dots=dots,
                is_rest=True,
            )
        )
        cursor += piece_slots
    return events


def decompose_duration_slots(duration_slots: int) -> list[tuple[int, str, int]]:
    remaining = duration_slots
    pieces: list[tuple[int, str, int]] = []
    for piece_slots, note_type, dots in DURATION_PIECES:
        while remaining >= piece_slots:
            pieces.append((piece_slots, note_type, dots))
            remaining -= piece_slots
    if remaining:
        pieces.append((1, "16th", 0))
    return pieces


def measure_voices_fill(measure: ScoreMeasure) -> bool:
    for voice_index in range(1, measure.voice_count + 1):
        voice_total = sum(event.duration_slots for event in measure.events if event.voice == voice_index)
        if voice_total != measure.total_slots:
            return False
    return True


def format_measure_ranges(measures: list[int]) -> str:
    if not measures:
        return ""

    ranges: list[str] = []
    start = measures[0]
    previous = measures[0]

    for measure in measures[1:]:
        if measure == previous + 1:
            previous = measure
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = measure
        previous = measure

    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)
