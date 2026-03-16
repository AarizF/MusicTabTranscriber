from pathlib import Path

from xml.etree import ElementTree as ET

import guitarpro

from backend.src.exporters import export_gp5, export_midi, export_musicxml, export_pdf
from backend.src.models import ConfidenceSummary, JobOptions, NoteEvent, TabEvent
from backend.src.score_pipeline import build_score_part


def test_exporters_create_files(tmp_path: Path) -> None:
    notes = [
        NoteEvent(onset_sec=0.0, offset_sec=0.5, midi_pitch=64, confidence=0.9),
        NoteEvent(onset_sec=0.5, offset_sec=1.0, midi_pitch=67, confidence=0.8),
    ]
    tabs = [
        TabEvent(onset_sec=0.0, offset_sec=0.5, midi_pitch=64, string=1, fret=0, confidence=0.9),
        TabEvent(onset_sec=0.5, offset_sec=1.0, midi_pitch=67, string=2, fret=8, confidence=0.8),
    ]
    summary = ConfidenceSummary(
        average_confidence=0.85,
        note_count=2,
        low_confidence_count=0,
        branch_used="original",
    )

    pdf_path = tmp_path / "sample.pdf"
    midi_path = tmp_path / "sample.mid"
    gp5_path = tmp_path / "sample.gp5"
    musicxml_path = tmp_path / "sample.musicxml"

    export_pdf(tabs, summary, pdf_path)
    export_midi(notes, midi_path)
    export_gp5(tabs, gp5_path, "Sample")
    score_part, warnings = build_score_part(
        "Sample",
        tabs,
        "unused.wav",
        1.0,
        JobOptions(tempo_override_bpm=120.0),
    )
    assert warnings == []
    assert score_part is not None
    export_musicxml(score_part, musicxml_path)

    assert pdf_path.exists()
    assert midi_path.exists()
    assert gp5_path.exists()
    assert musicxml_path.exists()

    parsed = guitarpro.parse(str(gp5_path))
    assert parsed.title == "Sample"
    assert len(parsed.tracks[0].measures[0].voices[0].beats) == 2

    xml_root = ET.fromstring(musicxml_path.read_text(encoding="utf-8"))
    assert xml_root.find(".//divisions") is not None
    assert xml_root.find(".//technical/string") is not None
    assert xml_root.find(".//technical/fret") is not None
    assert xml_root.find(".//staff-details") is not None
