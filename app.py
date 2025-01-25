from flask import Flask, request, render_template, url_for
import numpy as np
import cv2
import os
from keras.models import load_model
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask_cors import CORS
from waitress import serve

# Load environment variables
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', '0') == '1'
MODEL_PATH = os.environ.get('MODEL_PATH', 'Pneumonia_detection.h5')
CLASS_NAMES = ['Pneumonia', 'Normal']

# Load the trained model
model = load_model(MODEL_PATH)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to prepare the image for prediction
def prepare_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return "No file selected", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = prepare_image(filepath)
    predictions = model.predict(img)
    result = CLASS_NAMES[np.argmax(predictions)]

    return render_template(
        'result.html',
        prediction=result,
        name=request.form.get('name', 'Unknown'),
        age=request.form.get('age', 'Unknown'),
        sex=request.form.get('sex', 'Unknown'),
        image_path=url_for('static', filename=f'uploads/{filename}')
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    serve(app, host='0.0.0.0', port=port)
