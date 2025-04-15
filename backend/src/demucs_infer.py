import os
from demucs import pretrained
from demucs.apply import apply_model
from pathlib import Path
import torchaudio


def separate_guitar(input_file: str) -> str:
    """
    Separates the guitar track from the input audio file using Demucs.

    Args:
        input_file (str): Path to the input audio file (e.g., MP3 or WAV).

    Returns:
        str: Path to the isolated guitar track.
    """
    # Define the base data directory
    base_dir = Path(__file__).resolve().parent.parent.parent / "data"

    # Resolve full paths for input and output
    input_path = base_dir / input_file

    # Load the pre-trained Demucs model
    model = pretrained.get_model(name="htdemucs")
    model.cpu()  # Use CPU for inference (change to .cuda() if using GPU)

    # Load the input audio file
    wav, sr = torchaudio.load(input_path)
    wav = wav.mean(0).unsqueeze(0)  # Convert to mono if stereo

    # Apply the model to separate the audio
    sources = apply_model(model, wav, sr=sr, device="cpu", split=True)

    # Save only the "other" stem (assumed to contain the guitar)
    guitar_source = sources[1]  # [drums, bass, other, vocals]
    torchaudio.save(base_dir, guitar_source.cpu(), sr)

    # Return the path to the "other" stem (assumed to contain the guitar)
    guitar_path = base_dir / "other.wav"
    return str(guitar_path)
