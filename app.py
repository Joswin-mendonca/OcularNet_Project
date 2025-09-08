from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
import keras
keras.utils.get_custom_objects()

app = Flask(__name__)

# -----------------------------
# Config
# -----------------------------
UPLOAD_FOLDER = os.path.join("static", "uploads")  # save in static/uploads
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = "student_model_pruned_final_v2.keras"  # your trained model
CLASS_NAMES_PATH = "class_names.json"              # class names file
IMG_SIZE = (260, 260)                              # same as training size

# -----------------------------
# Load model and classes once
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

def preprocess_image(image_file):
    """Resize and preprocess uploaded image"""
    img = Image.open(image_file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/submit', methods=['POST'])
def submit():
    file = request.files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)

    # Preprocess & predict
    processed_img = preprocess_image(image_path)
    predictions = model.predict(processed_img)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    # Render result page with image + prediction
    return render_template(
        'result.html',
        result=f"{predicted_class} (confidence: {confidence:.2f})",
        filename=file.filename  # pass filename to template
    )

if __name__ == '__main__':
    app.run(debug=True)
