from pathlib import Path

from backend.src.models import JobOptions, TabEvent, TechniqueHint
from backend.src.score_pipeline import build_score_part, infer_beat_grid


def test_build_score_part_creates_measures_with_rests_and_ties(tmp_path: Path) -> None:
    audio_path = tmp_path / "dummy.wav"
    audio_path.write_bytes(b"RIFF")
    tab_events = [
        TabEvent(onset_sec=0.0, offset_sec=0.5, midi_pitch=52, string=6, fret=12, confidence=0.9),
        TabEvent(onset_sec=0.75, offset_sec=1.5, midi_pitch=55, string=5, fret=10, confidence=0.9),
        TabEvent(
            onset_sec=3.5,
            offset_sec=4.5,
            midi_pitch=57,
            string=5,
            fret=12,
            confidence=0.8,
            pitch_bend_cents=50,
            technique_hint=TechniqueHint.bend,
        ),
    ]
    options = JobOptions(tempo_override_bpm=120.0, time_signature_override="4/4")

    score_part, warnings = build_score_part("Riff", tab_events, str(audio_path), 5.0, options)

    assert warnings == []
    assert score_part is not None
    assert score_part.measure_fill_valid is True
    assert len(score_part.measures) >= 2
    assert any(event.is_rest for event in score_part.measures[0].events)
    assert any(
        note.tie_start or note.tie_stop
        for measure in score_part.measures
        for event in measure.events
        for note in event.notes
    )


def test_build_score_part_supports_two_voice_overlap_without_warning(tmp_path: Path) -> None:
    audio_path = tmp_path / "dummy.wav"
    audio_path.write_bytes(b"RIFF")
    tab_events = [
        TabEvent(onset_sec=0.0, offset_sec=1.0, midi_pitch=40, string=6, fret=0, confidence=0.9),
        TabEvent(onset_sec=0.5, offset_sec=0.75, midi_pitch=47, string=5, fret=2, confidence=0.9),
        TabEvent(onset_sec=0.75, offset_sec=1.0, midi_pitch=45, string=5, fret=0, confidence=0.9),
    ]
    options = JobOptions(tempo_override_bpm=120.0, time_signature_override="4/4")

    score_part, warnings = build_score_part("Pedal", tab_events, str(audio_path), 2.0, options)

    assert score_part is not None
    assert warnings == []
    assert score_part.measure_fill_valid is True
    assert score_part.measures[0].voice_count == 2
    assert {event.voice for event in score_part.measures[0].events if not event.is_rest} == {1, 2}


def test_infer_beat_grid_honors_manual_tempo_override() -> None:
    beat_grid = infer_beat_grid("unused.wav", 2.0, JobOptions(tempo_override_bpm=90.0))

    assert beat_grid.bpm == 90.0
    assert beat_grid.beat_duration_sec > 0
    assert beat_grid.subdivision_duration_sec == beat_grid.beat_duration_sec / 4
