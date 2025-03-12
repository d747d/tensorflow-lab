from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

app = Flask(__name__)

# Load the TensorFlow model (replace with your model path)
# model.save('iris_nn_model.h5')
#model = tf.keras.models.load_model('iris_nn_model.h5')
model = keras.applications.ResNet50V2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="resnet50v2",
)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input data from the form
        input_data = request.form['input_text']
        
        # Preprocess the input data (example: convert to numerical array)
        processed_data = preprocess_input(input_data)

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