from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model (you'll need to place your .h5 file in the models directory)
try:
    model = tf.keras.models.load_model('models/my_model.h5')
    print("Model loaded successfully!")
except:
    print("Model not found. Please place your .h5 file in the models directory.")
    model = None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize image to model input size (adjust dimensions as needed)
    image = image.resize((224, 224))
    # Convert to array
    image_array = np.array(image)
    # Normalize pixel values
    image_array = image_array / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/')
def predict_page():
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
       return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Read and preprocess image
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)
        class_idx = np.argmax(prediction[0])
        class_labels = ['healthy', 'moderate', 'severe']
        result = class_labels[class_idx]
        confidence = float(np.max(prediction[0]))

        return jsonify({
            'result': result,
            'confidence': f"{confidence * 100:.1f}%",
            'probabilities': prediction[0].tolist()
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Railway provides PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)