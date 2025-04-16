import os
from demucs import pretrained
from demucs.apply import apply_model
from pathlib import Path
import torchaudio

# Set the audio backend to sox_io
# torchaudio.set_audio_backend("sox_io")
# Install pip install torchaudio soundfile instead


def separate_guitar(input_path: str) -> str:
    """
    Separates the guitar track from the input audio file using Demucs.

    Args:
        input_path (str): Path to the input audio file (e.g., MP3 or WAV).

    Returns:
        str: Path to the isolated guitar track.
    """
    # Define the base data directory
    base_dir = Path(__file__).resolve().parent.parent.parent / "data"

    # Ensure the input file exists
    input_file = Path(input_path).resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # print(f"Input file path: {input_file}")
    # print(f"Torchaudio backend: {str(torchaudio.list_audio_backends())}")

    # Load the pre-trained Demucs model
    model = pretrained.get_model(name="htdemucs_6s")
    model.cpu()  # Use CPU for inference (change to .cuda() if using GPU)

    # Load the input audio file
    try:
        wav, sr = torchaudio.load(str(input_file))
        if wav.shape[0] == 1:  # If mono, convert to stereo
            wav = wav.repeat(2, 1)  # Duplicate the mono channel to create stereo
        wav = wav.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        raise RuntimeError(f"Error loading audio file: {input_file}. Details: {e}")

    # Apply the model to separate the audio
    try:
        sources = apply_model(model, wav, device="cpu", split=True)
    except Exception as e:
        raise RuntimeError(f"Error applying Demucs model: {e}")

    # Save the separated stems
    stem_type = ["drums", "bass", "1?", "vocals", "guitar", "2?"]
    for i, stem in enumerate(sources[0]):  # Iterate over stems
        stem_path = base_dir / f"stem_{stem_type[i]}.wav"
        torchaudio.save(stem_path, stem.cpu(), sr)
        print(f"Saved stem {i} to {stem_path}")

    # Return the path to the isolated guitar track
    return str(base_dir / "stem_guitar.wav")
