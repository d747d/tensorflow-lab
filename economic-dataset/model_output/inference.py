
import tensorflow as tf
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Function to download NLTK resources
def download_nltk_resources():
    for resource in ['punkt', 'stopwords', 'wordnet']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)

# Download resources
download_nltk_resources()

# Load model and resources
model = tf.keras.models.load_model('model_output/sentiment_model.h5')

with open('model_output/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
with open('model_output/thresholds.pickle', 'rb') as handle:
    thresholds = pickle.load(handle)

def preprocess_text(text):
    # Clean and preprocess text for sentiment analysis
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters, numbers, and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        words = nltk.word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)
    else:
        return ''

def predict_sentiment(text, max_len=150):
    # Predict sentiment for a given text.
    # Args:
    #     text: Text to analyze
    # Returns:
    #     Dictionary with sentiment prediction and details
    
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(padded)[0]
    pred_class = np.argmax(prediction)
    
    # Map to sentiment
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    result = {
        'sentiment': sentiment_map[pred_class],
        'confidence': float(prediction[pred_class]),
        'probabilities': {
            'Negative': float(prediction[0]),
            'Neutral': float(prediction[1]),
            'Positive': float(prediction[2])
        }
    }
    
    return result

# Example usage
if __name__ == "__main__":
    # Test with sample texts
    test_texts = [
        "Economic growth has surpassed expectations this quarter.",
        "Market shows signs of recession as indexes plummet.",
        "Financial stability maintained despite challenges."
    ]
    
    for text in test_texts:
        result = predict_sentiment(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: {result['probabilities']}")
