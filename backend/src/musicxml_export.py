from __future__ import annotations

from pathlib import Path
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

from .models import ScoreEvent, ScoreMeasure, ScorePart


TUNING = [
    ("E", 4),
    ("B", 3),
    ("G", 3),
    ("D", 3),
    ("A", 2),
    ("E", 2),
]


def export_musicxml(score_part: ScorePart, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xml_string = score_to_musicxml_string(score_part)
    output_path.write_text(xml_string, encoding="utf-8")


def score_to_musicxml_string(score_part: ScorePart) -> str:
    root = Element("score-partwise", version="4.0")
    work = SubElement(root, "work")
    SubElement(work, "work-title").text = score_part.title

    identification = SubElement(root, "identification")
    encoding = SubElement(identification, "encoding")
    SubElement(encoding, "software").text = "MusicTabTranscriber"

    part_list = SubElement(root, "part-list")
    score_part_el = SubElement(part_list, "score-part", id="P1")
    SubElement(score_part_el, "part-name").text = "Lead Guitar"
    SubElement(score_part_el, "part-abbreviation").text = "Gtr."
    SubElement(score_part_el, "score-instrument", id="P1-I1")
    midi_instrument = SubElement(score_part_el, "midi-instrument", id="P1-I1")
    SubElement(midi_instrument, "midi-channel").text = "1"
    SubElement(midi_instrument, "midi-program").text = "27"

    part = SubElement(root, "part", id="P1")
    for measure_index, measure in enumerate(score_part.measures):
        measure_el = SubElement(part, "measure", number=str(measure.number))
        if measure_index == 0:
            append_measure_attributes(measure_el, score_part)
            direction = SubElement(measure_el, "direction", placement="above")
            direction_type = SubElement(direction, "direction-type")
            metronome = SubElement(direction_type, "metronome")
            SubElement(metronome, "beat-unit").text = "quarter"
            SubElement(metronome, "per-minute").text = str(int(round(score_part.beat_grid.bpm)))
            sound = SubElement(direction, "sound")
            sound.set("tempo", f"{score_part.beat_grid.bpm:.2f}")

        write_measure_staff(measure_el, measure, staff_number=1)
        backup = SubElement(measure_el, "backup")
        SubElement(backup, "duration").text = str(measure.total_slots * measure.voice_count)
        write_measure_staff(measure_el, measure, staff_number=2)

    parsed = minidom.parseString(tostring(root, encoding="utf-8"))
    return parsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


def append_measure_attributes(measure_el: Element, score_part: ScorePart) -> None:
    attributes = SubElement(measure_el, "attributes")
    SubElement(attributes, "divisions").text = str(score_part.divisions)
    key = SubElement(attributes, "key")
    SubElement(key, "fifths").text = "0"
    time = SubElement(attributes, "time")
    beats, beat_type = score_part.time_signature.split("/")
    SubElement(time, "beats").text = beats
    SubElement(time, "beat-type").text = beat_type
    SubElement(attributes, "staves").text = "2"

    clef_notation = SubElement(attributes, "clef", number="1")
    SubElement(clef_notation, "sign").text = "G"
    SubElement(clef_notation, "line").text = "2"

    clef_tab = SubElement(attributes, "clef", number="2")
    SubElement(clef_tab, "sign").text = "TAB"
    SubElement(clef_tab, "line").text = "5"

    staff_details = SubElement(attributes, "staff-details", number="2")
    SubElement(staff_details, "staff-lines").text = "6"
    for index, (step, octave) in enumerate(TUNING, start=1):
        tuning = SubElement(staff_details, "staff-tuning", line=str(index))
        SubElement(tuning, "tuning-step").text = step
        SubElement(tuning, "tuning-octave").text = str(octave)


def write_measure_staff(measure_el: Element, measure: ScoreMeasure, staff_number: int) -> None:
    for voice_index in range(1, measure.voice_count + 1):
        voice_events = [event for event in measure.events if event.voice == voice_index]
        for event in voice_events:
            write_score_event(measure_el, event, staff_number)
        if voice_index < measure.voice_count:
            backup = SubElement(measure_el, "backup")
            SubElement(backup, "duration").text = str(measure.total_slots)


def write_score_event(measure_el: Element, event: ScoreEvent, staff_number: int) -> None:
    if event.is_rest:
        note_el = SubElement(measure_el, "note")
        SubElement(note_el, "rest")
        SubElement(note_el, "duration").text = str(event.duration_slots)
        SubElement(note_el, "voice").text = str(event.voice)
        SubElement(note_el, "type").text = event.note_type
        for _ in range(event.dots):
            SubElement(note_el, "dot")
        SubElement(note_el, "staff").text = str(staff_number)
        return

    for index, note in enumerate(event.notes):
        note_el = SubElement(measure_el, "note")
        if index > 0:
            SubElement(note_el, "chord")
        append_pitch(note_el, note.midi_pitch)
        if note.tie_stop:
            SubElement(note_el, "tie", type="stop")
        if note.tie_start:
            SubElement(note_el, "tie", type="start")
        SubElement(note_el, "duration").text = str(event.duration_slots)
        SubElement(note_el, "voice").text = str(event.voice)
        SubElement(note_el, "type").text = event.note_type
        for _ in range(event.dots):
            SubElement(note_el, "dot")
        SubElement(note_el, "staff").text = str(staff_number)

        notations = SubElement(note_el, "notations")
        if note.tie_stop:
            SubElement(notations, "tied", type="stop")
        if note.tie_start:
            SubElement(notations, "tied", type="start")
        technical = SubElement(notations, "technical")
        if note.string is not None:
            SubElement(technical, "string").text = str(note.string)
        if note.fret is not None:
            SubElement(technical, "fret").text = str(note.fret)
        if abs(note.pitch_bend_cents) >= 25:
            bend = SubElement(technical, "bend")
            SubElement(bend, "bend-alter").text = f"{note.pitch_bend_cents / 100.0:.2f}"


def append_pitch(note_el: Element, midi_pitch: int) -> None:
    names = [
        ("C", 0),
        ("C", 1),
        ("D", 0),
        ("D", 1),
        ("E", 0),
        ("F", 0),
        ("F", 1),
        ("G", 0),
        ("G", 1),
        ("A", 0),
        ("A", 1),
        ("B", 0),
    ]
    step, alter = names[midi_pitch % 12]
    octave = (midi_pitch // 12) - 1
    pitch = SubElement(note_el, "pitch")
    SubElement(pitch, "step").text = step
    if alter:
        SubElement(pitch, "alter").text = str(alter)
    SubElement(pitch, "octave").text = str(octave)
