from backend.src.evaluation_metrics import compute_note_metrics, compute_tab_metrics
from backend.src.models import NoteEvent, TabEvent


def test_compute_note_metrics_returns_perfect_scores_for_exact_match() -> None:
    reference = [
        NoteEvent(onset_sec=0.0, offset_sec=0.5, midi_pitch=52, confidence=1.0),
        NoteEvent(onset_sec=0.5, offset_sec=1.0, midi_pitch=55, confidence=1.0),
    ]
    predicted = [
        NoteEvent(onset_sec=0.0, offset_sec=0.5, midi_pitch=52, confidence=0.8),
        NoteEvent(onset_sec=0.5, offset_sec=1.0, midi_pitch=55, confidence=0.9),
    ]

    metrics = compute_note_metrics(reference, predicted)

    assert metrics["onset_f1"] == 1.0
    assert metrics["note_f1"] == 1.0
    assert metrics["pitch_accuracy"] == 1.0


def test_compute_tab_metrics_reports_string_and_fret_accuracy() -> None:
    reference = [
        TabEvent(onset_sec=0.0, offset_sec=0.5, midi_pitch=52, string=6, fret=12, confidence=1.0),
        TabEvent(onset_sec=0.5, offset_sec=1.0, midi_pitch=55, string=5, fret=10, confidence=1.0),
    ]
    predicted = [
        TabEvent(onset_sec=0.01, offset_sec=0.5, midi_pitch=52, string=6, fret=12, confidence=0.8),
        TabEvent(onset_sec=0.5, offset_sec=1.0, midi_pitch=55, string=4, fret=10, confidence=0.9),
    ]

    metrics = compute_tab_metrics(reference, predicted)

    assert metrics["string_accuracy"] == 0.5
    assert metrics["fret_accuracy"] == 1.0
    assert metrics["invalid_fingering_rate"] == 0.0
