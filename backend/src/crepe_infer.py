from __future__ import annotations

from .models import NoteEvent, TechniqueHint


def analyze_pitch(input_path: str) -> list[dict]:
    """
    Compatibility wrapper for older code paths.
    The rebuilt app uses Basic Pitch first, but this helper still exposes
    simple note dictionaries for anyone calling the legacy function.
    """
    from .transcription import transcribe_with_fallback

    events = transcribe_with_fallback(input_path, "original")
    return [
        {
            "time": event.onset_sec,
            "frequency": 440.0 * (2 ** ((event.midi_pitch - 69) / 12)),
            "note": midi_to_note_name(event.midi_pitch),
            "technique": TechniqueHint(event.technique_hint).value,
        }
        for event in events
    ]


def midi_to_note_name(midi_pitch: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi_pitch // 12) - 1
    return f"{names[midi_pitch % 12]}{octave}"
