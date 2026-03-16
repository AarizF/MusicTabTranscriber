from backend.src.models import BranchScoreBreakdown, JobOptions, NoteEvent
from backend.src.transcription import BranchTranscription, choose_best_branch, cleanup_note_events, fuse_branch_events


def test_cleanup_note_events_merges_fragments_and_suppresses_octave_ghosts() -> None:
    events = [
        NoteEvent(onset_sec=0.0, offset_sec=0.02, midi_pitch=52, confidence=0.4),
        NoteEvent(onset_sec=0.1, offset_sec=0.3, midi_pitch=52, confidence=0.8),
        NoteEvent(onset_sec=0.12, offset_sec=0.28, midi_pitch=64, confidence=0.45),
        NoteEvent(onset_sec=0.32, offset_sec=0.52, midi_pitch=52, confidence=0.7),
    ]

    cleaned, warnings = cleanup_note_events(events, lead_guitar_mode=True)

    assert len(cleaned) == 1
    assert cleaned[0].midi_pitch == 52
    assert cleaned[0].offset_sec >= 0.52
    assert any("ultra-short" in warning for warning in warnings)
    assert any("octave-ghost" in warning for warning in warnings)


def test_choose_best_branch_prefers_higher_overall_branch_score() -> None:
    original = BranchTranscription(
        branch_name="original",
        raw_note_events=[],
        note_events=[],
        average_confidence=0.6,
        warnings=[],
        cleanup_warnings=[],
        score=BranchScoreBreakdown(overall=0.55),
    )
    guitar_stem = BranchTranscription(
        branch_name="guitar_stem",
        raw_note_events=[],
        note_events=[],
        average_confidence=0.5,
        warnings=[],
        cleanup_warnings=[],
        score=BranchScoreBreakdown(overall=0.7),
    )

    chosen = choose_best_branch([original, guitar_stem])

    assert chosen.branch_name == "guitar_stem"


def test_fuse_branch_events_replaces_weaker_match_and_adds_strong_extra_note() -> None:
    primary = BranchTranscription(
        branch_name="original",
        raw_note_events=[],
        note_events=[
            NoteEvent(onset_sec=0.0, offset_sec=0.4, midi_pitch=52, confidence=0.55, source_branch="original"),
        ],
        average_confidence=0.55,
        warnings=[],
        cleanup_warnings=[],
        score=BranchScoreBreakdown(overall=0.6),
    )
    alternate = BranchTranscription(
        branch_name="guitar_stem",
        raw_note_events=[],
        note_events=[
            NoteEvent(onset_sec=0.01, offset_sec=0.42, midi_pitch=52, confidence=0.8, source_branch="guitar_stem"),
            NoteEvent(onset_sec=0.6, offset_sec=0.9, midi_pitch=55, confidence=0.82, source_branch="guitar_stem"),
        ],
        average_confidence=0.81,
        warnings=[],
        cleanup_warnings=[],
        score=BranchScoreBreakdown(overall=0.7),
    )

    fused, used = fuse_branch_events(primary, alternate, JobOptions(transcription_profile="accurate"))

    assert used is True
    assert len(fused) == 2
    assert fused[0].source_branch == "guitar_stem"
    assert any(event.midi_pitch == 55 for event in fused)
