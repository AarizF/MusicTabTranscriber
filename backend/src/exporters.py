from __future__ import annotations

from pathlib import Path

from .models import ArtifactKind, ConfidenceSummary, NoteEvent, ScorePart, TabEvent
from .musicxml_export import export_musicxml as write_musicxml
from .pdf_generator import save_tabs_to_pdf


def export_midi(note_events: list[NoteEvent], output_path: Path) -> None:
    try:
        import pretty_midi
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("pretty_midi is not installed.") from exc

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=26, name="Lead Guitar")
    for event in note_events:
        note = pretty_midi.Note(
            velocity=max(1, min(127, int(40 + event.confidence * 80))),
            pitch=event.midi_pitch,
            start=event.onset_sec,
            end=max(event.offset_sec, event.onset_sec + 0.05),
        )
        instrument.notes.append(note)
    midi.instruments.append(instrument)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))


def export_gp5(tab_events: list[TabEvent], output_path: Path, title: str) -> None:
    try:
        import guitarpro
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("PyGuitarPro is not installed.") from exc

    song = guitarpro.models.Song()
    song.title = title
    track = song.tracks[0]
    track.name = "Lead Guitar"

    measure = track.measures[0]
    voice = measure.voices[0]
    voice.beats = []
    for event in tab_events[:64]:
        beat = guitarpro.models.Beat(voice)
        beat.status = guitarpro.models.BeatStatus.normal
        beat.duration = guitarpro.models.Duration(value=4)
        note = guitarpro.models.Note(
            beat,
            value=event.fret,
            string=event.string,
            type=guitarpro.models.NoteType.normal,
        )
        if event.technique_hint == "bend" or getattr(event.technique_hint, "value", "") == "bend":
            note.effect.bend = guitarpro.models.BendEffect()
        beat.notes.append(note)
        voice.beats.append(beat)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    guitarpro.write(song, str(output_path))


def export_pdf(tab_events: list[TabEvent], summary: ConfidenceSummary, output_path: Path) -> None:
    save_tabs_to_pdf(tab_events, summary, str(output_path))


def export_musicxml(score_part: ScorePart, output_path: Path) -> None:
    write_musicxml(score_part, output_path)


def artifact_extension(kind: ArtifactKind) -> str:
    return {
        ArtifactKind.midi: ".mid",
        ArtifactKind.guitar_pro: ".gp5",
        ArtifactKind.pdf: ".pdf",
        ArtifactKind.musicxml: ".musicxml",
    }[kind]
