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
        
        # Optionally, save the processed text to a CSV for later model use
        processed_df = pd.DataFrame([{'original_text': raw_text, 'processed_text': preprocessed_text}])
        processed_file = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data.csv')
        processed_df.to_csv(processed_file, mode='a', header=not os.path.exists(processed_file), index=False)
        
        return f"File uploaded and processed. Preprocessed text: {preprocessed_text[:500]}...<br><a href='/upload'>Upload another file</a>"
    return "Invalid file or no file uploaded."

if __name__ == '__main__':
    app.run(debug=True)

