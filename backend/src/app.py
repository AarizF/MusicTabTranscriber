from flask import Flask, request, send_file
from pathlib import Path
from demucs_infer import separate_guitar
from crepe_infer import analyze_pitch
import subprocess
from datetime import datetime
import pytz

app = Flask(__name__)

# Define the base data directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "data"
BASE_DIR.mkdir(exist_ok=True)  # Ensure the data directory exists


def get_current_time_ist() -> str:
    """
    Returns the current time in IST (Indian Standard Time) as a string.
    """
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist).strftime("%H:%M:%S")


def convert_to_wav(input_path: Path) -> Path:
    """
    Converts an audio file to WAV format using ffmpeg, or reuses the existing WAV file if it already exists.

    Args:
        input_path (Path): Path to the input audio file.

    Returns:
        Path: Path to the converted WAV file.
    """
    output_path = input_path.with_suffix(".wav")

    # Check if the WAV file already exists
    if output_path.exists():
        print(f"[{get_current_time_ist()}] WAV file already exists: {output_path}")
        return output_path

    # Convert to WAV if the file does not exist
    try:
        subprocess.run(
            ["ffmpeg", "-i", str(input_path), str(output_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"[{get_current_time_ist()}] Conversion to WAV successful: {output_path}")
    except subprocess.CalledProcessError as e:
        print(
            f"[{get_current_time_ist()}] Error during conversion: {e.stderr.decode()}"
        )
        raise

    return output_path


@app.route("/transcribe", methods=["POST"])
def transcribe():
    # Get the uploaded file
    file = request.files["file"]

    # Save the file to the data directory
    file_path = BASE_DIR / file.filename
    file_path = Path(file_path).resolve()
    file.save(file_path)
    print(f"[{get_current_time_ist()}] File saved to: {file_path}")

    # Convert the file to WAV format
    wav_file_path = convert_to_wav(file_path)

    # Process the WAV file
    raw_music_data_path = separate_guitar(wav_file_path)
    music_notes = analyze_pitch(raw_music_data_path)

    # For now, return a dummy PDF file
    return send_file("dummy.pdf", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
