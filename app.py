from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import os
import cv2
import numpy as np
import base64
import io
from PIL import Image
import random
import tempfile
from src.explainability import get_gradcam_heatmap, overlay_heatmap

app = Flask(__name__)
CORS(app)

# Config
MODEL_PATH = 'model_output/visionspec_qc_v1.h5'
IMG_SIZE = (224, 224)

# Global Model Variable
model = None

def load_visionspec_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model Loaded Successfully.")
    else:
        print("Model File Not Found. Please run training first.")

# Initial model load
load_visionspec_model()

def encode_image_base64(img_np):
    """Converts an OpenCV image (BGR) to a Base64 string."""
    _, buffer = cv2.imencode('.png', img_np)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_b64}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    img_bytes = file.read()
    
    # Preprocess Image
    nparr = np.frombuffer(img_bytes, np.uint8)
    raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Save a temporary file (cross-platform)
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, 'temp_inspect.png')
    cv2.imwrite(temp_path, raw_img)
    
    img_rs = cv2.resize(raw_img, IMG_SIZE)
    img_array = np.expand_dims(img_rs, axis=0) / 255.0
    
    # 1. Prediction
    preds = model.predict(img_array, verbose=0)
    prediction = float(preds[0][0])
    label = "PASS" if prediction > 0.5 else "DEFECT"
    confidence = prediction if prediction > 0.5 else (1 - prediction)
    
    # 2. Explainability (Grad-CAM)
    heatmap = get_gradcam_heatmap(img_array, model, 'out_relu')
    overlay = overlay_heatmap(temp_path, heatmap)
    
    # 3. Base64 Encoding
    original_b64 = encode_image_base64(raw_img)
    overlay_b64 = encode_image_base64(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    return jsonify({
        'status': label,
        'confidence': round(confidence * 100, 2),
        'original_image': original_b64,
        'heatmap_image': overlay_b64
    })

@app.route('/simulation', methods=['GET'])
def simulation():
    """Returns a random image from the validation set for the 'live-feed' simulation demo."""
    splits = ['Pass', 'Defect']
    split = random.choice(splits)
    val_dir = os.path.join('data', 'val', split)
    files = [f for f in os.listdir(val_dir) if f.endswith('.bmp')]
    picked = random.choice(files)
    
    # Return path or better yet, base64
    img_path = os.path.join(val_dir, picked)
    img = cv2.imread(img_path)
    return jsonify({
        'filename': picked,
        'category': split,
        'image': encode_image_base64(img)
    })

import random # for simulation

if __name__ == '__main__':
    app.run(debug=True, port=5000)
