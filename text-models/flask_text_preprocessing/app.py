import os
import pandas as pd
import nltk
import gensim
from gensim import corpora
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pyLDAvis
import pyLDAvis.gensim_models

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
    
    # Debugging: Print out the first 10 tokens after cleaning
    print(f"Preprocessed Text (tokens): {tokens[:10]}")  # First 10 tokens
    
    return ' '.join(tokens)

# Dataset file path
DATASET_PATH = os.path.join(UPLOAD_FOLDER, 'dataset.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    
    if not files:
        return "No files uploaded."
    
    # Process each uploaded file
    for file in files:
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

            # Delete the original file after processing
            os.remove(filepath)
    
    return redirect('/topic_modeling')  # Redirect to topic modeling page

@app.route('/reset', methods=['POST'])
def reset_data():
    if os.path.exists(DATASET_PATH):
        os.remove(DATASET_PATH)  # Delete the dataset file
        return render_template('index.html', message="Dataset has been reset successfully.")
    else:
        return render_template('index.html', message="No dataset file found to reset.")

@app.route('/topic_modeling', methods=['GET', 'POST'])
def topic_modeling():
    if request.method == 'POST':
        # Load the dataset
        data = pd.read_csv(DATASET_PATH)

        # Ensure that the necessary column is present
        if 'processed_text' not in data.columns:
            return render_template('topic_modeling.html', message="No processed text column found.")

        # Preprocess the text (Tokenization, Remove stopwords, Remove punctuation)
        def preprocess_text(text):
            tokens = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
            return tokens

        # Apply preprocessing to the 'processed_text' column
        processed_docs = data['processed_text'].apply(preprocess_text)

        # Debugging: Print number of processed documents
        print(f"Number of Documents: {len(processed_docs)}")

        # Create a dictionary and a corpus
        dictionary = corpora.Dictionary(processed_docs)
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

        # Debugging: Print out the corpus
        print(f"Corpus: {corpus[:5]}")  # Check first 5 documents

        # Train the LDA model
        lda_model = gensim.models.LdaMulticore(corpus, num_topics=3, id2word=dictionary, passes=10, workers=4)

        # Check topics
        topics = lda_model.print_topics(num_words=5)
        print(f"Topics: {topics}")  # Check generated topics

        # Prepare topics for display
        topic_list = [f"Topic {i}: {topic[1]}" for i, topic in enumerate(topics)]

        # Visualize the topics using pyLDAvis
        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        vis_html = pyLDAvis.prepared_data_to_html(vis)

        return render_template('topic_modeling.html', topics=topic_list, vis_html=vis_html)
    return render_template('topic_modeling.html')

if __name__ == '__main__':
    app.run(debug=True)
