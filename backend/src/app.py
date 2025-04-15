from flask import Flask, request, send_file
import time

app = Flask(__name__)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    file = request.files["file"]
    # Process the MP3 file and generate a PDF
    print("before sleep")
    time.sleep(5)
    print("after sleep")
    # For now, return a dummy PDF file
    return send_file("dummy.pdf", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
