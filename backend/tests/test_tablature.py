from backend.src.models import JobOptions, NoteEvent, TechniqueHint
from backend.src.tab_generator import generate_tab_events


def test_sequence_solver_returns_playable_positions() -> None:
    notes = [
        NoteEvent(onset_sec=0.0, offset_sec=0.4, midi_pitch=64, confidence=0.9),
        NoteEvent(onset_sec=0.45, offset_sec=0.8, midi_pitch=66, confidence=0.8),
        NoteEvent(
            onset_sec=0.9,
            offset_sec=1.2,
            midi_pitch=69,
            confidence=0.7,
            pitch_bend_cents=40,
            technique_hint=TechniqueHint.bend,
        ),
    ]

    tab_events = generate_tab_events(notes, JobOptions(max_fret=20))

    assert len(tab_events) == 3
    assert all(0 <= event.fret <= 20 for event in tab_events)
    assert all(1 <= event.string <= 6 for event in tab_events)


def test_solver_skips_out_of_range_notes() -> None:
    notes = [
        NoteEvent(onset_sec=0.0, offset_sec=0.4, midi_pitch=20, confidence=0.9),
        NoteEvent(onset_sec=0.5, offset_sec=0.8, midi_pitch=64, confidence=0.8),
    ]

    tab_events = generate_tab_events(notes, JobOptions(max_fret=20))

    assert len(tab_events) == 1
    assert tab_events[0].midi_pitch == 64


def test_solver_recovers_when_an_intermediate_event_has_no_valid_transition() -> None:
    notes = [
        NoteEvent(onset_sec=0.0, offset_sec=0.5, midi_pitch=40, confidence=0.9),
        NoteEvent(onset_sec=0.2, offset_sec=0.7, midi_pitch=41, confidence=0.9),
        NoteEvent(onset_sec=0.8, offset_sec=1.1, midi_pitch=42, confidence=0.9),
    ]

    tab_events = generate_tab_events(notes, JobOptions(max_fret=20))

    assert len(tab_events) == 3
    assert [event.fret for event in tab_events] == [0, 1, 2]
    assert [event.string for event in tab_events] == [1, 1, 1]
