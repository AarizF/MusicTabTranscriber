from flask import Flask, request, send_file
from pathlib import Path
from demucs_infer import separate_guitar

app = Flask(__name__)

# Define the base data directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "data"
BASE_DIR.mkdir(exist_ok=True)  # Ensure the data directory exists


@app.route("/transcribe", methods=["POST"])
def transcribe():
    # Get the uploaded file
    file = request.files["file"]

    # Save the file to the data directory
    file_path = BASE_DIR / file.filename
    file.save(file_path)
    print(f"File saved to: {file_path}")

    # Process the MP3 file
    raw_music_data_path = separate_guitar(file.filename)

    # For now, return a dummy PDF file
    return send_file("dummy.pdf", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
