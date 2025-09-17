"""
Lightweight Plant Disease Predictor
Optimized version that can use quantized TensorFlow Lite models for faster inference
"""

import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
import json
from datetime import datetime
from pathlib import Path


class LightweightPlantDiseasePredictor:
    def __init__(self, model_path='optimized_models/model_quantized_float16.tflite', 
                 class_names_path='class_names.txt', fallback_model='best_model.h5'):
        """
        Initialize the lightweight plant disease predictor
        
        Args:
            model_path (str): Path to the optimized model (TFLite or .h5)
            class_names_path (str): Path to the class names file
            fallback_model (str): Fallback model if optimized model not available
        """
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.fallback_model = fallback_model
        self.model = None
        self.interpreter = None
        self.class_names = []
        self.model_type = 'unknown'
        self.is_quantized = False
        self.input_dtype = np.float32
        self.IMG_HEIGHT = 224
        self.IMG_WIDTH = 224
        
        # Disease information database
        self.disease_info = self._load_disease_info()
        
        # Load model and class names
        self.load_model()
        self.load_class_names()
    
    def _load_disease_info(self):
        """Load disease information database"""
        return {
            'Apple___Apple_scab': {
                'severity': 'Moderate',
                'description': 'Fungal disease causing dark spots on leaves and fruit',
                'treatment': 'Apply fungicides, remove infected leaves, improve air circulation',
                'prevention': 'Plant resistant varieties, avoid overhead watering'
            },
            'Apple___Cedar_apple_rust': {
                'severity': 'Moderate',
                'description': 'Fungal disease causing orange spots on leaves',
                'treatment': 'Remove nearby cedar trees, apply fungicides in spring',
                'prevention': 'Plant resistant varieties, maintain distance from cedar trees'
            },
            'Tomato___Early_blight': {
                'severity': 'High',
                'description': 'Fungal disease causing brown spots with concentric rings',
                'treatment': 'Apply copper-based fungicides, remove affected leaves',
                'prevention': 'Rotate crops, mulch soil, avoid overhead watering'
            },
            'Tomato___Late_blight': {
                'severity': 'Critical',
                'description': 'Highly destructive fungal disease, can kill entire plants',
                'treatment': 'Remove infected plants immediately, apply fungicides preventively',
                'prevention': 'Use resistant varieties, ensure good ventilation, avoid wet conditions'
            },
            'Potato___Early_blight': {
                'severity': 'Moderate',
                'description': 'Fungal disease causing dark spots on leaves',
                'treatment': 'Apply fungicides, remove infected foliage',
                'prevention': 'Rotate crops, plant certified seed potatoes'
            },
            'Potato___Late_blight': {
                'severity': 'Critical',
                'description': 'Devastating disease that caused Irish Potato Famine',
                'treatment': 'Destroy infected plants, apply preventive fungicides',
                'prevention': 'Use resistant varieties, avoid planting in wet conditions'
            },
            # Add more diseases as needed
        }
    
    def load_model(self):
        """Load the optimized model with fallback support"""
        try:
            # Check if it's a TensorFlow Lite model
            if self.model_path.endswith('.tflite') and os.path.exists(self.model_path):
                self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
                self.interpreter.allocate_tensors()
                
                # Get input and output details
                input_details = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()
                
                self.input_details = input_details[0]
                self.output_details = output_details[0]
                
                # Check if model is quantized
                self.input_dtype = self.input_details['dtype']
                self.is_quantized = self.input_dtype == np.uint8
                
                # Get input shape
                input_shape = self.input_details['shape']
                self.IMG_HEIGHT = input_shape[1]
                self.IMG_WIDTH = input_shape[2]
                
                if 'int8' in str(self.model_path).lower():
                    self.model_type = 'tflite_int8'
                elif 'float16' in str(self.model_path).lower():
                    self.model_type = 'tflite_float16'
                else:
                    self.model_type = 'tflite'
                
                print(f"✅ TensorFlow Lite model loaded from {self.model_path}")
                print(f"📊 Model type: {self.model_type}")
                print(f"🎯 Input shape: {input_shape}")
                print(f"🔢 Input dtype: {self.input_dtype}")
                print(f"⚡ Quantized: {self.is_quantized}")
                
            # Try regular Keras model
            elif os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                
                # Check the actual input shape
                input_shape = self.model.input_shape
                self.IMG_HEIGHT = input_shape[1]
                self.IMG_WIDTH = input_shape[2]
                
                if 'lightweight' in str(self.model_path).lower():
                    self.model_type = 'lightweight'
                else:
                    self.model_type = 'keras'
                
                print(f"✅ Keras model loaded from {self.model_path}")
                print(f"📊 Model type: {self.model_type}")
                print(f"🧠 Parameters: {self.model.count_params():,}")
                
            # Fallback to original model
            elif os.path.exists(self.fallback_model):
                self.model = tf.keras.models.load_model(self.fallback_model, compile=False)
                self.model_type = 'fallback'
                self.IMG_HEIGHT = 224
                self.IMG_WIDTH = 224
                print(f"⚠️ Using fallback model from {self.fallback_model}")
                
            else:
                raise FileNotFoundError("No model file found")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_class_names(self):
        """Load class names from file"""
        try:
            with open(self.class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            print(f"✅ Loaded {len(self.class_names)} class names")
        except Exception as e:
            print(f"❌ Error loading class names: {e}")
            raise
    
    def preprocess_image_for_tflite(self, image_array):
        """
        Preprocess image for TensorFlow Lite model
        """
        # Ensure correct shape
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        # Resize if needed
        if image_array.shape[1] != self.IMG_HEIGHT or image_array.shape[2] != self.IMG_WIDTH:
            img = Image.fromarray((image_array[0] * 255).astype(np.uint8))
            img = img.resize((self.IMG_WIDTH, self.IMG_HEIGHT), Image.Resampling.LANCZOS)
            image_array = np.expand_dims(np.array(img).astype(np.float32) / 255.0, axis=0)
        
        # Convert to appropriate data type for quantized models
        if self.is_quantized:
            # For INT8 quantized models, convert to uint8
            image_array = (image_array * 255).astype(np.uint8)
        else:
            # For float models, keep as float32
            image_array = image_array.astype(np.float32)
        
        return image_array
    
    def predict_with_tflite(self, image_array):
        """
        Make prediction using TensorFlow Lite interpreter
        """
        # Preprocess image for TFLite
        processed_image = self.preprocess_image_for_tflite(image_array)
        
        # Set input
        self.interpreter.set_tensor(self.input_details['index'], processed_image)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        predictions = self.interpreter.get_tensor(self.output_details['index'])
        
        return predictions[0]  # Remove batch dimension
    
    def predict_with_keras(self, image_array):
        """
        Make prediction using Keras model
        """
        # Ensure proper shape and type
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        # Normalize if needed
        if image_array.max() > 1.0:
            image_array = image_array.astype('float32') / 255.0
        
        # Resize if needed
        if image_array.shape[1] != self.IMG_HEIGHT or image_array.shape[2] != self.IMG_WIDTH:
            img = Image.fromarray((image_array[0] * 255).astype(np.uint8))
            img = img.resize((self.IMG_WIDTH, self.IMG_HEIGHT), Image.Resampling.LANCZOS)
            image_array = np.expand_dims(np.array(img).astype('float32') / 255.0, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image_array, verbose=0)
        return predictions[0]  # Remove batch dimension
    
    def predict_image_from_array(self, image_array, top_n=5, use_tta=False):
        """
        Make prediction on an image array (optimized for web uploads)
        
        Args:
            image_array (np.array): Image array
            top_n (int): Number of top predictions to return
            use_tta (bool): Whether to use test-time augmentation (disabled for lightweight models)
            
        Returns:
            dict: Comprehensive prediction results
        """
        try:
            # Choose prediction method based on model type
            if self.interpreter is not None:
                predictions = self.predict_with_tflite(image_array)
            else:
                predictions = self.predict_with_keras(image_array)
            
            # Get top N predictions
            top_indices = np.argsort(predictions)[-top_n:][::-1]
            
            results = []
            for idx in top_indices:
                class_name = self.class_names[idx]
                confidence = float(predictions[idx])
                results.append((class_name, confidence))
            
            # Format comprehensive results
            formatted_results = self.format_comprehensive_results(results, use_tta, False)
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            raise
    
    def format_comprehensive_results(self, results, used_tta=False, enhanced_image=False):
        """
        Format prediction results with comprehensive information
        """
        if not results:
            return {'error': 'No predictions available'}
        
        top_prediction = results[0]
        top_class = top_prediction[0]
        top_confidence = top_prediction[1]
        
        # Parse class name
        parts = top_class.split('___')
        plant = parts[0] if len(parts) > 0 else 'Unknown'
        disease = parts[1] if len(parts) > 1 else 'Unknown'
        
        # Get disease information
        disease_data = self.disease_info.get(top_class, {
            'severity': 'Unknown',
            'description': 'Information not available',
            'treatment': 'Consult agricultural expert',
            'prevention': 'Follow general plant care guidelines'
        })
        
        # Determine confidence level
        confidence_level = 'High' if top_confidence > 0.8 else 'Medium' if top_confidence > 0.5 else 'Low'
        
        formatted_results = {
            'top_prediction': top_class,
            'confidence': top_confidence,
            'confidence_percentage': f"{top_confidence * 100:.2f}%",
            'confidence_level': confidence_level,
            'plant': plant,
            'disease': disease,
            'is_healthy': 'healthy' in disease.lower(),
            'model_type': self.model_type,
            'used_tta': used_tta,
            'enhanced_image': enhanced_image,
            'timestamp': datetime.now().isoformat(),
            'disease_info': disease_data,
            'all_predictions': []
        }
        
        # Add all predictions with detailed info
        for class_name, confidence in results:
            parts = class_name.split('___')
            pred_plant = parts[0] if len(parts) > 0 else 'Unknown'
            pred_disease = parts[1] if len(parts) > 1 else 'Unknown'
            
            pred_info = {
                'plant': pred_plant,
                'disease': pred_disease,
                'full_name': class_name,
                'confidence': confidence,
                'confidence_percentage': f"{confidence * 100:.2f}%",
                'is_healthy': 'healthy' in pred_disease.lower()
            }
            
            formatted_results['all_predictions'].append(pred_info)
        
        return formatted_results
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.interpreter is not None:
            model_size = os.path.getsize(self.model_path) / (1024 * 1024)  # MB
            return {
                'model_type': self.model_type,
                'model_path': self.model_path,
                'input_size': f"{self.IMG_WIDTH}x{self.IMG_HEIGHT}",
                'num_classes': len(self.class_names),
                'supports_tta': False,  # Disabled for lightweight models
                'is_quantized': self.is_quantized,
                'model_size_mb': f"{model_size:.2f}",
                'inference_optimized': True
            }
        else:
            model_size = os.path.getsize(self.model_path) / (1024 * 1024) if os.path.exists(self.model_path) else 0
            return {
                'model_type': self.model_type,
                'model_path': self.model_path,
                'input_size': f"{self.IMG_WIDTH}x{self.IMG_HEIGHT}",
                'num_classes': len(self.class_names),
                'supports_tta': self.model_type == 'keras',
                'is_quantized': False,
                'model_size_mb': f"{model_size:.2f}",
                'inference_optimized': self.model_type == 'lightweight'
            }


class ModelSelector:
    """
    Utility class to automatically select the best available model
    """
    
    @staticmethod
    def get_best_available_model():
        """
        Automatically select the best available optimized model
        Priority: Float16 TFLite > INT8 TFLite > Lightweight Keras > Original
        """
        models_to_try = [
            ('optimized_models/model_quantized_float16.tflite', 'float16_tflite'),
            ('optimized_models/model_quantized_int8.tflite', 'int8_tflite'),
            ('optimized_models/model_lightweight_architecture.h5', 'lightweight_keras'),
            ('best_model.h5', 'original')
        ]
        
        for model_path, model_name in models_to_try:
            if os.path.exists(model_path):
                print(f"🎯 Selected model: {model_name} ({model_path})")
                return model_path, model_name
        
        raise FileNotFoundError("No suitable model found")
    
    @staticmethod
    def create_optimized_predictor():
        """
        Create a predictor with the best available optimized model
        """
        try:
            model_path, model_name = ModelSelector.get_best_available_model()
            predictor = LightweightPlantDiseasePredictor(model_path=model_path)
            
            print(f"✅ Created optimized predictor with {model_name}")
            print(f"📊 Model info: {predictor.get_model_info()}")
            
            return predictor
            
        except Exception as e:
            print(f"❌ Failed to create optimized predictor: {e}")
            # Fallback to original predictor
            from predict_advanced import AdvancedPlantDiseasePredictor
            return AdvancedPlantDiseasePredictor()


# Test function
def test_lightweight_predictor():
    """Test the lightweight predictor"""
    try:
        predictor = ModelSelector.create_optimized_predictor()
        
        # Create a dummy test image
        test_image = np.random.random((224, 224, 3)).astype(np.float32)
        
        print("\n🧪 Testing prediction with dummy image...")
        results = predictor.predict_image_from_array(test_image, top_n=3)
        
        print(f"🎯 Top prediction: {results['plant']} - {results['disease']}")
        print(f"📈 Confidence: {results['confidence_percentage']}")
        print(f"🏥 Severity: {results['disease_info']['severity']}")
        print(f"⚡ Model type: {results['model_type']}")
        
        return predictor
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None


if __name__ == "__main__":
    test_lightweight_predictor()