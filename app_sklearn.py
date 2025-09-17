"""
Flask Web Application for Plant Disease Classification using Scikit-learn
This app loads a pickled scikit-learn model and provides image upload and prediction functionality
"""

import os
import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
from PIL import Image
import logging
import time
import json
from feature_extractor import PlantDiseaseFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
model_data = None
predictor = None

class SklearnPlantDiseasePredictor:
    """Scikit-learn based plant disease predictor"""
    
    def __init__(self, model_path):
        """Initialize the predictor with a trained model"""
        self.model_path = model_path
        self.model = None
        self.feature_extractor = None
        self.label_encoder = None
        self.class_names = None
        self.model_info = {}
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and components"""
        try:
            logger.info(f"🔄 Loading scikit-learn model from {self.model_path}")
            
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_extractor = model_data['feature_extractor']
            self.label_encoder = model_data['label_encoder']
            self.class_names = model_data['class_names']
            
            self.model_info = {
                'model_name': model_data.get('model_name', 'Unknown'),
                'test_accuracy': model_data.get('test_accuracy', 0.0),
                'num_classes': len(self.class_names),
                'model_type': 'scikit-learn',
                'features': 'HOG + Color Histogram + LBP + Texture'
            }
            
            logger.info(f"✅ Model loaded successfully!")
            logger.info(f"📊 Model: {self.model_info['model_name']}")
            logger.info(f"🎯 Accuracy: {self.model_info['test_accuracy']:.4f}")
            logger.info(f"📊 Classes: {self.model_info['num_classes']}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """Preprocess image and extract features"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract features using the same feature extractor from training
            features = self.feature_extractor.extract_features(image)
            
            return features.reshape(1, -1)  # Reshape for single prediction
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing image: {e}")
            raise
    
    def predict(self, image_path, top_k=5):
        """Predict plant disease from image"""
        try:
            start_time = time.time()
            
            # Preprocess image
            features = self.preprocess_image(image_path)
            
            # Get predictions
            probabilities = self.model.predict_proba(features)[0]
            predicted_class_idx = self.model.predict(features)[0]
            
            # Get top k predictions
            top_indices = np.argsort(probabilities)[::-1][:top_k]
            
            predictions = []
            for i, idx in enumerate(top_indices):
                class_name = self.class_names[idx]
                confidence = float(probabilities[idx])
                
                predictions.append({
                    'rank': i + 1,
                    'class': class_name,
                    'confidence': confidence,
                    'percentage': confidence * 100
                })
            
            # Calculate inference time
            inference_time = time.time() - start_time
            
            result = {
                'success': True,
                'predictions': predictions,
                'top_prediction': {
                    'class': self.class_names[predicted_class_idx],
                    'confidence': float(probabilities[predicted_class_idx]),
                    'percentage': float(probabilities[predicted_class_idx]) * 100
                },
                'inference_time_ms': inference_time * 1000,
                'model_info': self.model_info
            }
            
            logger.info(f"🎯 Prediction: {result['top_prediction']['class']} ({result['top_prediction']['percentage']:.2f}%)")
            logger.info(f"⏱️ Inference time: {inference_time*1000:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_info': self.model_info
            }
    
    def get_model_info(self):
        """Get model information"""
        return self.model_info

def load_predictor():
    """Load the scikit-learn predictor"""
    global predictor
    
    model_files = [
        'plant_disease_sklearn_model.pkl',
        'plant_disease_sklearn_model_joblib.pkl'
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                predictor = SklearnPlantDiseasePredictor(model_file)
                logger.info(f"✅ Predictor loaded from {model_file}")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to load {model_file}: {e}")
                continue
    
    logger.error("❌ No valid model file found!")
    return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page"""
    model_info = predictor.get_model_info() if predictor else {}
    return render_template('index_sklearn.html', model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload an image.'})
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predictor.predict(filepath)
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Error in predict route: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'model_info': predictor.get_model_info() if predictor else {}
    })

@app.route('/benchmark')
def benchmark():
    """Benchmark the model performance"""
    if not predictor:
        return jsonify({'success': False, 'error': 'Model not loaded'})
    
    return jsonify({
        'success': True,
        'model_info': predictor.get_model_info(),
        'benchmark_info': {
            'note': 'Scikit-learn model with feature extraction',
            'features': 'HOG + Color Histogram + LBP + Texture',
            'preprocessing': 'Image resize + feature extraction',
            'model_size': 'Variable (depends on training data)'
        }
    })

# Create the HTML template
template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KrishiVannai AI - Plant Disease Detection</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #10b981;
            --primary-dark: #059669;
            --secondary-color: #3b82f6;
            --background: #f8fafc;
            --surface: #ffffff;
            --text-primary: #1f2937;
            --text-secondary: #6b7280;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --border: #e5e7eb;
            --shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            --border-radius: 12px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #10b981 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem;
            line-height: 1.6;
        }
        
        .container {
            background: var(--surface);
            padding: 2.5rem;
            border-radius: 24px;
            box-shadow: var(--shadow);
            max-width: 700px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .header {
            text-align: center;
            margin-bottom: 2.5rem;
        }
        
        .header h1 {
            color: var(--text-primary);
            margin-bottom: 0.5rem;
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header p {
            color: var(--text-secondary);
            font-size: 1.1rem;
            font-weight: 400;
        }
        
        .model-badge {
            background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 50px;
            font-size: 0.9rem;
            font-weight: 600;
            margin: 1.5rem 0;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }
        
        .model-info {
            background: linear-gradient(135deg, #f8fafc, #f1f5f9);
            padding: 1.5rem;
            border-radius: var(--border-radius);
            margin: 1.5rem 0;
            border: 1px solid var(--border);
        }
        
        .model-info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .info-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }
        
        .info-item i {
            color: var(--primary-color);
            width: 16px;
        }
        
        .info-value {
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .upload-area {
            border: 2px dashed var(--border);
            border-radius: var(--border-radius);
            padding: 3rem 2rem;
            text-align: center;
            margin: 2rem 0;
            transition: var(--transition);
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        
        .upload-area::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.05), rgba(59, 130, 246, 0.05));
            opacity: 0;
            transition: var(--transition);
        }
        
        .upload-area:hover::before,
        .upload-area.dragover::before {
            opacity: 1;
        }
        
        .upload-area:hover {
            border-color: var(--primary-color);
            transform: translateY(-2px);
        }
        
        .upload-area.dragover {
            border-color: var(--primary-color);
            background: rgba(16, 185, 129, 0.05);
        }
        
        .upload-icon {
            font-size: 3rem;
            color: var(--primary-color);
            margin-bottom: 1rem;
        }
        
        .upload-text {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }
        
        .upload-subtext {
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }
        
        .upload-formats {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .format-badge {
            background: var(--background);
            color: var(--text-secondary);
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            border: 1px solid var(--border);
        }
        
        #fileInput {
            display: none;
        }
        
        .loading {
            text-align: center;
            display: none;
            margin: 2rem 0;
            padding: 2rem;
        }
        
        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid var(--border);
            border-top: 3px solid var(--primary-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        
        .loading-text {
            color: var(--text-secondary);
            font-weight: 500;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .results {
            display: none;
            margin-top: 2rem;
        }
        
        .results-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            font-size: 1.3rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .prediction-card {
            background: var(--surface);
            border-radius: var(--border-radius);
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid var(--border);
            transition: var(--transition);
            position: relative;
            overflow: hidden;
        }
        
        .prediction-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(135deg, var(--success), var(--primary-color));
        }
        
        .prediction-card.top {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.05), rgba(16, 185, 129, 0.02));
            border-color: var(--success);
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
        }
        
        .prediction-card.top .disease-name {
            color: var(--success);
        }
        
        .prediction-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
        }
        
        .disease-name {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .confidence-percentage {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--success);
        }
        
        .confidence-bar {
            background: var(--background);
            height: 8px;
            border-radius: 4px;
            margin: 1rem 0;
            overflow: hidden;
            position: relative;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--success), var(--primary-color));
            border-radius: 4px;
            transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }
        
        .confidence-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .error {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05));
            color: var(--error);
            padding: 1.5rem;
            border-radius: var(--border-radius);
            margin: 1rem 0;
            border: 1px solid rgba(239, 68, 68, 0.2);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .inference-time {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 1.5rem;
            padding: 1rem;
            background: var(--background);
            border-radius: var(--border-radius);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 1.5rem;
                margin: 0.5rem;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .upload-area {
                padding: 2rem 1rem;
            }
            
            .model-info-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-seedling"></i> KrishiVannai AI</h1>
            <p>Advanced Plant Disease Detection System</p>
            <div class="model-badge">
                <i class="fas fa-brain"></i>
                Scikit-learn Model
            </div>
            
            {% if model_info %}
            <div class="model-info">
                <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 1rem;">
                    <i class="fas fa-info-circle"></i> Model Information
                </div>
                <div class="model-info-grid">
                    <div class="info-item">
                        <i class="fas fa-cogs"></i>
                        <span>Algorithm: <span class="info-value">{{ model_info.model_name }}</span></span>
                    </div>
                    <div class="info-item">
                        <i class="fas fa-chart-line"></i>
                        <span>Accuracy: <span class="info-value">{{ "%.1f%%"|format(model_info.test_accuracy * 100) }}</span></span>
                    </div>
                    <div class="info-item">
                        <i class="fas fa-layer-group"></i>
                        <span>Classes: <span class="info-value">{{ model_info.num_classes }}</span></span>
                    </div>
                    <div class="info-item">
                        <i class="fas fa-vector-square"></i>
                        <span>Features: <span class="info-value">{{ model_info.features }}</span></span>
                    </div>
                </div>
            </div>
            {% endif %}
        </div>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">
                <i class="fas fa-cloud-upload-alt"></i>
            </div>
            <div class="upload-text">Upload Plant Image</div>
            <div class="upload-subtext">Click here or drag and drop an image for analysis</div>
            <div class="upload-formats">
                <span class="format-badge">JPG</span>
                <span class="format-badge">PNG</span>
                <span class="format-badge">GIF</span>
                <span class="format-badge">BMP</span>
                <span class="format-badge">WebP</span>
            </div>
            <input type="file" id="fileInput" accept="image/*">
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div class="loading-text">
                <i class="fas fa-microscope pulse"></i>
                Analyzing plant image...
            </div>
        </div>
        
        <div class="results" id="results"></div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.querySelector('.upload-area');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');

        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                showError('Please select an image file.');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            loading.style.display = 'block';
            results.style.display = 'none';

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';
                if (data.success) {
                    displayResults(data);
                } else {
                    showError(data.error || 'Prediction failed');
                }
            })
            .catch(error => {
                loading.style.display = 'none';
                showError('Network error: ' + error.message);
            });
        }

        function displayResults(data) {
            let html = `
                <div class="results-header">
                    <i class="fas fa-stethoscope"></i>
                    Diagnosis Results
                </div>
            `;
            
            // Top prediction
            const top = data.top_prediction;
            html += `
                <div class="prediction-card top">
                    <div class="prediction-header">
                        <div class="disease-name">
                            <i class="fas fa-leaf"></i>
                            ${top.class}
                        </div>
                        <div class="confidence-percentage">${top.percentage.toFixed(1)}%</div>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: 0%; animation-delay: 0.5s;" 
                             data-width="${top.percentage}"></div>
                    </div>
                    <div style="font-size: 0.9rem; color: var(--text-secondary); margin-top: 0.5rem;">
                        <i class="fas fa-shield-alt"></i> Primary Diagnosis
                    </div>
                </div>
            `;
            
            // Alternative predictions
            if (data.predictions && data.predictions.length > 1) {
                html += `
                    <div style="margin: 1.5rem 0 1rem; font-weight: 600; color: var(--text-primary);">
                        <i class="fas fa-list-alt"></i> Alternative Diagnoses
                    </div>
                `;
                data.predictions.slice(1, 4).forEach((pred, index) => {
                    html += `
                        <div class="prediction-card">
                            <div class="prediction-header">
                                <div class="disease-name">${pred.class}</div>
                                <div class="confidence-percentage" style="color: var(--text-secondary);">
                                    ${pred.percentage.toFixed(1)}%
                                </div>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: 0%; animation-delay: ${1 + index * 0.2}s;" 
                                     data-width="${pred.percentage}"></div>
                            </div>
                        </div>
                    `;
                });
            }
            
            // Inference time
            if (data.inference_time_ms) {
                html += `
                    <div class="inference-time">
                        <i class="fas fa-stopwatch"></i>
                        Analysis completed in ${data.inference_time_ms.toFixed(1)}ms
                    </div>
                `;
            }
            
            results.innerHTML = html;
            results.style.display = 'block';
            
            // Animate confidence bars
            setTimeout(() => {
                document.querySelectorAll('.confidence-fill[data-width]').forEach(bar => {
                    bar.style.width = bar.getAttribute('data-width') + '%';
                });
            }, 100);
        }

        function showError(message) {
            results.innerHTML = `
                <div class="error">
                    <i class="fas fa-exclamation-triangle"></i>
                    ${message}
                </div>
            `;
            results.style.display = 'block';
        }
    </script>
</body>
</html>'''

# Save the template
if not os.path.exists('templates'):
    os.makedirs('templates')

with open('templates/index_sklearn.html', 'w', encoding='utf-8') as f:
    f.write(template_content)

if __name__ == '__main__':
    print("🌱 KrishiVannai AI - Plant Disease Detection (Scikit-learn Version)")
    print("=" * 60)
    
    # Load the predictor
    if not load_predictor():
        print("❌ Failed to load model. Please ensure the model file exists.")
        exit(1)
    
    print("🚀 Starting Flask application...")
    print(f"📊 Model: {predictor.get_model_info()['model_name']}")
    print(f"🎯 Accuracy: {predictor.get_model_info()['test_accuracy']:.4f}")
    print(f"📊 Classes: {predictor.get_model_info()['num_classes']}")
    print("🌐 Server will start at http://127.0.0.1:5002")
    
    app.run(debug=True, host='127.0.0.1', port=5002)