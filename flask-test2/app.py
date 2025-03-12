from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow import keras
#from keras.models import load_model
import os
import keras
import numpy as np
from keras import layers
import string
import re

os.environ["KERAS_BACKEND"] = "tensorflow"
# Having looked at our data above, we see that the raw text contains HTML break
# tags of the form '<br />'. These tags will not be removed by the default
# standardizer (which doesn't strip HTML). Because of this, we will need to
# create a custom standardization function.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )


# Model constants.
max_features = 20000
embedding_dim = 128
sequence_length = 500

# Now that we have our custom standardization, we can instantiate our text
# vectorization layer. We are using this layer to normalize, split, and map
# strings to integers, so we set our 'output_mode' to 'int'.
# Note that we're using the default split function,
# and the custom standardization defined above.
# We also set an explicit maximum sequence length, since the CNNs later in our
# model won't support ragged sequences.
vectorize_layer = keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

# Now that the vectorize_layer has been created, call `adapt` on a text-only
# dataset to create the vocabulary. You don't have to batch, but for very large
# datasets this means you're not keeping spare copies of the dataset in memory.

# Let's make a text-only dataset (no labels):
text_ds = raw_train_ds.map(lambda x, y: x)
# Let's call `adapt`:
vectorize_layer.adapt(text_ds)


app = Flask(__name__)



# Load the TensorFlow model (replace with your model path)
# model.save('iris_nn_model.h5')
#model = tf.keras.models.load_model('iris_nn_model.h5')
batch_size = 32
raw_train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=1337,
)
raw_val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=1337,
)
raw_test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
)

print(f"Number of batches in raw_train_ds: {raw_train_ds.cardinality()}")
print(f"Number of batches in raw_val_ds: {raw_val_ds.cardinality()}")
print(f"Number of batches in raw_test_ds: {raw_test_ds.cardinality()}")

for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(5):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])
        
        

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input data from the form
        input_data = request.form['input_text']
        text_input = keras.Input(shape=(1,), dtype=tf.string, name='text')
        x = vectorize_layer(input_data)
        x = layers.Embedding(max_features + 1, embedding_dim)(x)
        
        
        # Preprocess the input data (example: convert to numerical array)
        processed_data = preprocess_input(text_input)

        # Make prediction using the TensorFlow model
        prediction = model.predict(processed_data)
        
        # Post-process the prediction (example: convert to readable output)
        output = postprocess_prediction(prediction)

        return render_template('index.html', output=output)
    return render_template('index.html')

def preprocess_input(input_text):
    # Implement your input preprocessing logic here
    # Convert the input text to numerical data suitable for your model
    # Example: Tokenization, padding, etc.
    return processed_data

def postprocess_prediction(prediction):
    # Implement your prediction post-processing logic here
    # Convert the model's output to a human-readable format
    return output

if __name__ == '__main__':
    app.run(debug=True)