import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import os

# Function to download NLTK resources
def download_nltk_resources():
    for resource in ['punkt', 'stopwords', 'wordnet']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)

# Text preprocessing function for efficient transformation
def preprocess_text(text):
    """Clean and preprocess text for sentiment analysis"""
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

# Main function for sentiment analysis pipeline
def sentiment_analysis_pipeline(csv_path, output_dir='model_output'):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download NLTK resources
    download_nltk_resources()
    
    # 1. Load and explore data
    print("Loading and exploring data...")
    df = pd.read_csv(csv_path, encoding='iso-8859-1')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Handle missing values
    df['headline'] = df['headline'].fillna('')
    df['text'] = df['text'].fillna('')
    
    # 2. Preprocess data
    print("Preprocessing text data...")
    # Apply preprocessing to headline and text
    df['headline_processed'] = df['headline'].apply(preprocess_text)
    df['text_processed'] = df['text'].apply(preprocess_text)
    
    # Combine preprocessed text
    df['content'] = df['headline_processed'] + ' ' + df['text_processed']
    
    # Analyze positivity distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['positivity'], bins=20)
    plt.title('Distribution of Positivity Scores')
    plt.xlabel('Positivity Score')
    plt.ylabel('Frequency')
    plt.savefig(f'{output_dir}/positivity_distribution.png')
    
    # 3. Define sentiment categories based on data distribution
    q1 = df['positivity'].quantile(0.33)
    q2 = df['positivity'].quantile(0.67)
    
    df['sentiment'] = pd.cut(
        df['positivity'], 
        bins=[-float('inf'), q1, q2, float('inf')], 
        labels=[0, 1, 2]  # 0: negative, 1: neutral, 2: positive
    )
    
    # Convert to numeric
    df['sentiment'] = df['sentiment'].astype(int)
    
    print("Sentiment distribution:")
    sentiment_counts = df['sentiment'].value_counts().sort_index()
    print(sentiment_counts)
    
    # Save the thresholds
    thresholds = {'q1': q1, 'q2': q2}
    with open(f'{output_dir}/thresholds.pickle', 'wb') as handle:
        pickle.dump(thresholds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # 4. Split data
    X = df['content'].values
    y = df['sentiment'].values
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 5. Tokenize text
    max_features = 15000
    max_len = 150
    
    tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    # Convert to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')
    
    # 6. Calculate class weights for imbalanced data
    class_counts = df['sentiment'].value_counts().sort_index()
    total = class_counts.sum()
    class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    print("Class weights:", class_weights)
    
    # 7. Build CNN-LSTM model for sentiment analysis
    print("Building and training model...")
    model = Sequential([
        # Embedding layer
        Embedding(max_features, 128, input_length=max_len),
        
        # CNN layers for feature extraction
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(5),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(5),
        
        # LSTM layer for sequence processing
        Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
        
        # Dense layers
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # 3 classes
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    checkpoint = ModelCheckpoint(
        f'{output_dir}/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Train model
    history = model.fit(
        X_train_pad, y_train,
        validation_data=(X_val_pad, y_val),
        epochs=15,
        batch_size=64,
        class_weight=class_weights,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    # 8. Evaluate model
    print("Evaluating model...")
    # Load best model
    model = tf.keras.models.load_model(f'{output_dir}/best_model.h5')
    
    # Evaluate on test set
    loss, accuracy = model.evaluate(X_test_pad, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test_pad)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Classification report
    report = classification_report(y_test, y_pred_classes, target_names=['Negative', 'Neutral', 'Positive'])
    print("Classification Report:")
    print(report)
    
    # Save classification report
    with open(f'{output_dir}/classification_report.txt', 'w') as f:
        f.write(report)
    
    # 9. Visualize results
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = tf.math.confusion_matrix(y_test, y_pred_classes).numpy()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    
    # Training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_history.png')
    
    # 10. Save model and tokenizer
    model.save(f'{output_dir}/sentiment_model.h5')
    with open(f'{output_dir}/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Model saved to {output_dir}/sentiment_model.h5")
    print(f"Tokenizer saved to {output_dir}/tokenizer.pickle")
    
    # 11. Create prediction function
    def predict_sentiment(text, model=model, tokenizer=tokenizer, max_len=max_len):
        """
        Predict sentiment for a given text.
        Args:
            text: Text to analyze
        Returns:
            Dictionary with sentiment prediction and details
        """
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
    
    # 12. Test with examples
    example_texts = [
        "Economic indicators show strong recovery in the manufacturing sector.",
        "Unemployment rate increases to 7.5%, worse than expected.",
        "Markets remain stable despite global uncertainties."
    ]
    
    print("\nExample Predictions:")
    for text in example_texts:
        prediction = predict_sentiment(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {prediction['sentiment']}")
        print(f"Confidence: {prediction['confidence']:.4f}")
    
    # 13. Create and save inference script
    inference_script = """
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
    """Clean and preprocess text for sentiment analysis"""
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
    """
    Predict sentiment for a given text.
    Args:
        text: Text to analyze
    Returns:
        Dictionary with sentiment prediction and details
    """
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
        print(f"\\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: {result['probabilities']}")
    """
    
    # Save inference script
    with open(f'{output_dir}/inference.py', 'w') as f:
        f.write(inference_script)
    
    print(f"Inference script saved to {output_dir}/inference.py")
    
    return model, tokenizer, predict_sentiment

# Run the sentiment analysis pipeline
if __name__ == "__main__":
    model, tokenizer, predict_sentiment = sentiment_analysis_pipeline('FullEconomicNewsDFE839861.csv')
