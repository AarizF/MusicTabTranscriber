from __future__ import annotations

from dataclasses import dataclass
from math import inf

from .models import JobOptions, NoteEvent, TabEvent, TechniqueHint


DEFAULT_TUNING = [40, 45, 50, 55, 59, 64]


@dataclass(frozen=True)
class Position:
    string: int
    fret: int


def candidate_positions(midi_pitch: int, tuning: list[int], max_fret: int) -> list[Position]:
    candidates: list[Position] = []
    for index, open_pitch in enumerate(tuning, start=1):
        fret = midi_pitch - open_pitch
        if 0 <= fret <= max_fret:
            candidates.append(Position(string=index, fret=fret))
    return sorted(candidates, key=lambda item: (item.fret, item.string))


def transition_cost(
    previous_event: NoteEvent,
    previous_position: Position,
    current_event: NoteEvent,
    current_position: Position,
    prefer_lower_positions: bool,
) -> float:
    if current_event.onset_sec < previous_event.offset_sec and current_position.string == previous_position.string:
        return inf

    string_jump = abs(current_position.string - previous_position.string)
    fret_jump = abs(current_position.fret - previous_position.fret)
    position_shift_penalty = max(0, fret_jump - 4) * 0.35
    overlap_penalty = 1.5 if current_event.onset_sec < previous_event.offset_sec else 0.0
    bend_penalty = 1.0 if current_event.technique_hint == TechniqueHint.bend and current_position.fret < 2 else 0.0
    lower_fret_bonus = current_position.fret * (0.03 if prefer_lower_positions else 0.0)

    return string_jump * 0.8 + fret_jump * 0.55 + position_shift_penalty + overlap_penalty + bend_penalty + lower_fret_bonus


def generate_tab_events(note_events: list[NoteEvent], options: JobOptions) -> list[TabEvent]:
    ordered_events = sorted(note_events, key=lambda item: (item.onset_sec, item.midi_pitch))
    if not ordered_events:
        return []

    tuning = options.tuning or DEFAULT_TUNING
    usable_pairs = [
        (event, candidate_positions(event.midi_pitch, tuning, options.max_fret))
        for event in ordered_events
    ]
    usable_pairs = [(event, positions) for event, positions in usable_pairs if positions]
    if not usable_pairs:
        return []

    states: list[dict[Position, float]] = []
    parents: list[dict[Position, Position | None]] = []

    for index, (event, event_candidates) in enumerate(usable_pairs):
        current_costs: dict[Position, float] = {}
        current_parents: dict[Position, Position | None] = {}
        for position in event_candidates:
            base_cost = position.fret * (0.06 if options.prefer_lower_positions else 0.0)
            if index == 0 or not states:
                current_costs[position] = base_cost
                current_parents[position] = None
                continue

            best_cost = inf
            best_parent = None
            previous_costs = states[-1]
            previous_event = usable_pairs[index - 1][0]
            for previous_position, previous_total in previous_costs.items():
                candidate_cost = previous_total + transition_cost(
                    previous_event,
                    previous_position,
                    event,
                    position,
                    options.prefer_lower_positions,
                )
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_parent = previous_position

            current_costs[position] = best_cost + base_cost
            current_parents[position] = best_parent

        states.append(current_costs)
        parents.append(current_parents)

    if not states:
        return []

    final_position = min(states[-1], key=states[-1].get)
    chosen_positions = [final_position]
    for index in range(len(parents) - 1, 0, -1):
        parent = parents[index][chosen_positions[-1]]
        if parent is not None:
            chosen_positions.append(parent)
    chosen_positions.reverse()

    tab_events: list[TabEvent] = []
    for (event, _), position in zip(usable_pairs, chosen_positions):
        tab_events.append(
            TabEvent(
                onset_sec=event.onset_sec,
                offset_sec=event.offset_sec,
                midi_pitch=event.midi_pitch,
                string=position.string,
                fret=position.fret,
                confidence=event.confidence,
                pitch_bend_cents=event.pitch_bend_cents,
                technique_hint=event.technique_hint,
                source_branch=event.source_branch,
            )
        )

    return tab_events


def generate_tabs(music_notes: list[dict]) -> str:
    """
    Compatibility wrapper that renders text tabs from legacy note dictionaries.
    """
    notes = [
        NoteEvent(
            onset_sec=float(item["time"]),
            offset_sec=float(item["time"]) + 0.25,
            midi_pitch=note_name_to_midi(item["note"]),
            confidence=float(item.get("confidence", 0.5)),
        )
        for item in music_notes
        if item.get("note")
    ]
    tab_events = generate_tab_events(notes, JobOptions())
    return render_ascii_tab(tab_events)


def render_ascii_tab(tab_events: list[TabEvent], columns: int = 64) -> str:
    lines = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    for event in tab_events[:columns]:
        for string_number in lines:
            lines[string_number].append(str(event.fret) if event.string == string_number else "-")
    names = {6: "E", 5: "A", 4: "D", 3: "G", 2: "B", 1: "e"}
    return "\n".join(
        f"{names[string_number]}| {' '.join(lines[string_number])}"
        for string_number in (1, 2, 3, 4, 5, 6)
    )


def note_name_to_midi(note_name: str) -> int:
    note_map = {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
    }
    pitch = note_name[:-1]
    octave = int(note_name[-1])
    return 12 * (octave + 1) + note_map[pitch]
