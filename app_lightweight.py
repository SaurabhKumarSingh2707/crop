"""
KrishiVannai AI Plant Disease Prediction System - Lightweight Version
Optimized Flask web application using quantized models for faster inference
"""

from flask import Flask, request, render_template, jsonify
import os
import numpy as np
from PIL import Image
import io
import base64
from predict_lightweight import LightweightPlantDiseasePredictor, ModelSelector
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'krishivannai-ai-plant-disease-prediction-lightweight'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size (reduced from 32MB)

# Global error handler for 500 errors only
@app.errorhandler(500)
def handle_internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}\n{traceback.format_exc()}")
    return jsonify({
        'success': False,
        'error': 'Internal server error occurred'
    }), 500

# Handle favicon requests
@app.route('/favicon.ico')
def favicon():
    """Return empty response for favicon requests"""
    return '', 204

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the lightweight predictor
predictor = None

def initialize_predictor():
    """Initialize predictor with optimized models"""
    global predictor
    try:
        predictor = ModelSelector.create_optimized_predictor()
        logger.info("✅ Lightweight predictor initialized successfully")
        
        # Log model info
        model_info = predictor.get_model_info()
        logger.info(f"📊 Using model: {model_info['model_type']}")
        logger.info(f"📏 Model size: {model_info['model_size_mb']} MB")
        logger.info(f"⚡ Inference optimized: {model_info['inference_optimized']}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize lightweight predictor: {e}")
        logger.error(traceback.format_exc())
        return False

# Try to initialize predictor
initialize_predictor()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_uploaded_image(file):
    """
    Process uploaded image and convert to format suitable for prediction
    
    Args:
        file: Uploaded file object
        
    Returns:
        tuple: (processed_image_array, original_image_base64, image_info)
    """
    try:
        # Read image
        image = Image.open(file.stream)
        
        # Get image info
        image_info = {
            'format': image.format,
            'mode': image.mode,
            'size': image.size,
            'filename': file.filename
        }
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save original image as base64 for display
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG', quality=85)  # Reduced quality for faster processing
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        original_image_b64 = f"data:image/jpeg;base64,{img_str}"
        
        # Resize for prediction
        target_size = (predictor.IMG_WIDTH, predictor.IMG_HEIGHT) if predictor else (224, 224)
        resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
        image_array = np.array(resized_image)
        image_array = image_array.astype('float32') / 255.0
        
        return image_array, original_image_b64, image_info
    
    except Exception as e:
        logger.error(f"❌ Error processing image: {e}")
        raise

@app.route('/')
def index():
    """Enhanced home page with lightweight model info"""
    try:
        logger.info(f"Index route accessed. Predictor status: {predictor is not None}")
        
        if predictor is None:
            logger.error("Predictor is None - attempting to reinitialize")
            if not initialize_predictor():
                model_info = {'model_type': 'unavailable', 'error': 'Failed to initialize predictor'}
            else:
                model_info = predictor.get_model_info()
        else:
            model_info = predictor.get_model_info()
            
        logger.info(f"Model info: {model_info}")
        return render_template('index_advanced.html', model_info=model_info)
        
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        logger.error(traceback.format_exc())
        return f"<h1>Error Loading Page</h1><p>{str(e)}</p><p>Predictor Status: {predictor is not None}</p>", 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and lightweight prediction"""
    try:
        # Check if predictor is available
        if not predictor:
            logger.error("Predictor not available - attempting to reinitialize")
            if not initialize_predictor():
                return jsonify({
                    'success': False,
                    'error': 'Prediction service unavailable. Model not loaded.'
                }), 503
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload PNG, JPG, JPEG, GIF, BMP, TIFF, or WebP files.'
            }), 400
        
        # Get prediction options from form (simplified for lightweight version)
        top_n = min(int(request.form.get('top_n', 5)), 10)  # Max 10 predictions
        
        # Process image
        try:
            image_array, original_image_b64, image_info = process_uploaded_image(file)
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return jsonify({
                'success': False,
                'error': f'Error processing image: {str(e)}'
            }), 400
        
        # Make prediction
        try:
            start_time = datetime.now()
            results = predictor.predict_image_from_array(
                image_array, 
                top_n=top_n, 
                use_tta=False  # Disabled for lightweight models
            )
            end_time = datetime.now()
            
            # Calculate inference time
            inference_time = (end_time - start_time).total_seconds() * 1000  # ms
            
            # Add image and processing info to results
            results['original_image'] = original_image_b64
            results['image_info'] = image_info
            results['processing_options'] = {
                'use_tta': False,
                'enhance_image': False,
                'top_n': top_n
            }
            results['inference_time_ms'] = round(inference_time, 2)
            results['model_optimized'] = True
            
            return jsonify({
                'success': True,
                'results': results
            })
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return jsonify({
                'success': False,
                'error': f'Error making prediction: {str(e)}'
            }), 500
    
    except Exception as e:
        logger.error(f"Unexpected error in predict: {e}")
        return jsonify({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch prediction for multiple images (optimized)"""
    try:
        if not predictor:
            return jsonify({'error': 'Prediction service unavailable'}), 503
        
        files = request.files.getlist('files')
        if not files or len(files) == 0:
            return jsonify({'error': 'No files uploaded'}), 400
        
        if len(files) > 5:  # Reduced batch size for lightweight processing
            return jsonify({'error': 'Maximum 5 files allowed in batch mode'}), 400
        
        results = []
        
        for file in files:
            if file.filename == '' or not allowed_file(file.filename):
                continue
            
            try:
                image_array, original_image_b64, image_info = process_uploaded_image(file)
                prediction = predictor.predict_image_from_array(image_array, top_n=3, use_tta=False)
                
                prediction['original_image'] = original_image_b64
                prediction['image_info'] = image_info
                results.append(prediction)
                
            except Exception as e:
                results.append({
                    'error': f'Failed to process {file.filename}: {str(e)}',
                    'filename': file.filename
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'processed_count': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.route('/model_info')
def model_info():
    """Get detailed model information"""
    if not predictor:
        return jsonify({'error': 'Predictor not available'}), 503
    
    info = predictor.get_model_info()
    info['classes'] = predictor.class_names[:10]  # First 10 classes
    info['total_classes'] = len(predictor.class_names)
    
    return jsonify(info)

@app.route('/health')
def health_check():
    """Enhanced health check endpoint for lightweight version"""
    health_status = {
        'status': 'healthy' if predictor else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'predictor_available': predictor is not None,
        'model_loaded': True if predictor else False,
        'classes_loaded': len(predictor.class_names) if predictor else 0,
        'version': 'lightweight',
        'optimized': True
    }
    
    if predictor:
        model_info = predictor.get_model_info()
        health_status.update({
            'model_type': model_info['model_type'],
            'model_size_mb': model_info['model_size_mb'],
            'inference_optimized': model_info['inference_optimized'],
            'is_quantized': model_info['is_quantized']
        })
    
    status_code = 200 if predictor else 503
    return jsonify(health_status), status_code

@app.route('/benchmark')
def benchmark():
    """Benchmark endpoint to test inference speed"""
    if not predictor:
        return jsonify({'error': 'Predictor not available'}), 503
    
    try:
        # Create a dummy test image
        test_image = np.random.random((224, 224, 3)).astype(np.float32)
        
        # Run multiple predictions to get average time
        times = []
        for _ in range(10):
            start_time = datetime.now()
            predictor.predict_image_from_array(test_image, top_n=1)
            end_time = datetime.now()
            times.append((end_time - start_time).total_seconds() * 1000)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        fps = 1000 / avg_time
        
        return jsonify({
            'average_inference_time_ms': round(avg_time, 2),
            'min_inference_time_ms': round(min_time, 2),
            'max_inference_time_ms': round(max_time, 2),
            'approximate_fps': round(fps, 1),
            'model_info': predictor.get_model_info()
        })
        
    except Exception as e:
        return jsonify({'error': f'Benchmark failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting KrishiVannai AI Plant Disease Prediction App (Lightweight Version)...")
    
    if predictor:
        model_info = predictor.get_model_info()
        print(f"📊 Model Type: {model_info['model_type']}")
        print(f"🧠 Model Path: {model_info['model_path']}")
        print(f"📐 Input Size: {model_info['input_size']}")
        print(f"🎯 Classes: {model_info['num_classes']}")
        print(f"📏 Model Size: {model_info['model_size_mb']} MB")
        print(f"⚡ Optimized: {model_info['inference_optimized']}")
        print(f"🔢 Quantized: {model_info['is_quantized']}")
    else:
        print("⚠️ Predictor not available - check model files")
    
    print(f"🌐 Starting lightweight server on port 5001")  # Different port to avoid conflicts
    app.run(debug=True, host='127.0.0.1', port=5001)