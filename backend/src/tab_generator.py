from typing import List, Dict, Optional, Tuple

# Map string number to its open string note in MIDI
# string 6 (low E) -> string 1 (high E)
STRING_TUNING = {
    6: 40,  # E2
    5: 45,  # A2
    4: 50,  # D3
    3: 55,  # G3
    2: 59,  # B3
    1: 64,  # E4
}

NOTE_TO_SEMITONE = {
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


def note_to_midi(note: str) -> int:
    """Convert note like 'F2' to MIDI number."""
    pitch = note[:-1]
    octave = int(note[-1])
    semitone = NOTE_TO_SEMITONE[pitch]
    return 12 * (octave + 1) + semitone


def find_string_and_fret(note: str, max_fret: int = 20) -> Optional[Tuple[int, int]]:
    """Return the best string/fret pair for a note like 'F2'."""
    note_midi = note_to_midi(note)
    options = []

    for string, open_midi in STRING_TUNING.items():
        fret = note_midi - open_midi
        if 0 <= fret <= max_fret:
            options.append((string, fret))

    if not options:
        return None  # Out of range

    # Prefer lower strings (thicker), so reverse sort
    options.sort(key=lambda x: (x[1], -x[0]))  # lower fret, thicker string preferred
    return options[0]


def generate_tabs(music_notes: List[Dict]) -> str:
    """
    Generates guitar tabs from music notes.

    Args:
        music_notes (List[Dict]): {'time': np.float64(0.57), 'frequency': np.float64(43.182239701905985), 'note': 'G#0'}

    Returns:
        str: Formatted guitar tabs as a string.
    """
    # Initialize the tab lines for 6 guitar strings
    strings = ["e", "B", "G", "D", "A", "E"]  # String names (high e to low E)
    tab_lines = {string: [] for string in strings}

    # Process each note and add it to the correct string
    for note in music_notes:
        time = int(note["time"] * 100)  # Convert time to 1/100s
        note_name = note["note"]

        # Find the string and fret for the note
        string_and_fret = find_string_and_fret(note_name)
        if string_and_fret is None:
            continue  # Skip notes that are out of range

        string_number, fret = string_and_fret
        string_name = strings[6 - string_number]  # Map string number to string name

        # Add the note to the correct string at the correct time
        for s in strings:
            if s == string_name:
                tab_lines[s].append(str(fret))
            else:
                tab_lines[s].append("-")

    # Format the tab lines into a readable string
    tab_output = ""
    for string in strings:
        tab_output += f"{string}| " + " ".join(tab_lines[string]) + "\n"

    return tab_output
