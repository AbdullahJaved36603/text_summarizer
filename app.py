from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import rag_pipeline
from parser_utils import parse_file  

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'md'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    all_texts = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            extracted = parse_file(file_path)
            all_texts.append(extracted)

    rag_pipeline.create_vector_store(all_texts)
    return jsonify({"status": "success", "message": "Files uploaded and processed."})

@app.route('/summarize', methods=['POST'])
def summarize():
    query = request.json.get('query', "Summarize this document")
    summary = rag_pipeline.summarize_with_rag(query)
    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True)
