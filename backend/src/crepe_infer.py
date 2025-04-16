import crepe
import numpy as np
from pathlib import Path
import torchaudio
import torch


def analyze_pitch(input_path: str) -> list[dict]:
    """
    Analyzes the pitch and notes from the input audio file using CREPE.

    Args:
        input_file (str): Path to the input audio file (e.g., WAV).

    Returns:
        list[dict]: A list of dictionaries containing time, frequency, and note information.
    """
    # Load the audio file
    audio, sr = torchaudio.load(input_path)
    audio = audio.mean(0).numpy()  # Convert to mono if stereo

    # Ensure the sample rate is 16 kHz (CREPE requirement)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(
            torch.tensor(audio)
        ).numpy()
        sr = 16000

    # Analyze pitch using CREPE
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)

    # Map frequencies to musical notes
    results = []
    for t, f, c in zip(time, frequency, confidence):
        if c > 0.5:  # Only consider confident predictions
            note = frequency_to_note(f)
            results.append({"time": t, "frequency": f, "note": note})

    print(results[0:100])  # Print the first 100 results for debugging

    return results


def frequency_to_note(frequency: float) -> str:
    """
    Converts a frequency to the nearest musical note.

    Args:
        frequency (float): Frequency in Hz.

    Returns:
        str: The corresponding musical note (e.g., A4, C#5).
    """
    if frequency <= 0:
        return "N/A"

    # Reference frequency for A4
    A4 = 440.0
    semitones = 12 * np.log2(frequency / A4)
    note_index = int(round(semitones)) % 12
    octave = int(np.floor(np.log2(frequency / A4) + 4))
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{notes[note_index]}{octave}"
