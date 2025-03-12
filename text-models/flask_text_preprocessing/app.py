import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import nltk
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize Flask app
app = Flask(__name__)

# Set allowed file types and upload folder
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'txt'}

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Set the upload folder and allowed file extensions for Flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocessing functions
def clean_text(text):
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Dataset file path
DATASET_PATH = os.path.join(UPLOAD_FOLDER, 'dataset.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read and process the file
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        # Clean the text (preprocessing)
        preprocessed_text = clean_text(raw_text)

        # Prepare data for dataset
        data = {
            'filename': filename,
            'original_text': raw_text,
            'processed_text': preprocessed_text
        }

        # Check if dataset file exists, create if not
        if not os.path.exists(DATASET_PATH):
            df = pd.DataFrame([data])
            df.to_csv(DATASET_PATH, mode='w', header=True, index=False)
        else:
            df = pd.DataFrame([data])
            df.to_csv(DATASET_PATH, mode='a', header=False, index=False)

        # Return response with an upload form
        return render_template('upload_again.html', preprocessed_text=preprocessed_text[:500])
    return "Invalid file or no file uploaded."

@app.route('/upload_again', methods=['GET'])
def upload_again():
    return render_template('index.html')

@app.route('/reset', methods=['POST'])
def reset_data():
    if os.path.exists(DATASET_PATH):
        os.remove(DATASET_PATH)
        return render_template('index.html', message="Dataset has been reset successfully.")
    return render_template('index.html', message="No dataset file found to reset.")

if __name__ == '__main__':
    app.run(debug=True)
