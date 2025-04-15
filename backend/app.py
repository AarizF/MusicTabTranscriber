from flask import Flask, request, send_file
import time

app = Flask(__name__)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    file = request.files['file']
    # Process the MP3 file and generate a PDF
    time.sleep(5)
    # For now, return a dummy PDF file
    return send_file('dummy.pdf', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)