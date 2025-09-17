import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
import json
from datetime import datetime

class AdvancedPlantDiseasePredictor:
    def __init__(self, model_path='best_model_advanced.h5', class_names_path='class_names_advanced.txt', fallback_model='best_model.h5'):
        """
        Initialize the advanced plant disease predictor with fallback support
        
        Args:
            model_path (str): Path to the advanced trained model
            class_names_path (str): Path to the class names file
            fallback_model (str): Fallback model if advanced model not available
        """
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.fallback_model = fallback_model
        self.model = None
        self.class_names = []
        self.model_type = 'unknown'
        self.IMG_HEIGHT = 300  # Default for advanced model
        self.IMG_WIDTH = 300
        
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
        """Load the trained model with fallback support"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                
                # Check the actual input shape to determine model type
                input_shape = self.model.input_shape
                if input_shape[1] == 300 and input_shape[2] == 300:
                    self.model_type = 'advanced'
                    self.IMG_HEIGHT = 300
                    self.IMG_WIDTH = 300
                    print(f"✅ Advanced model loaded from {self.model_path} (300x300)")
                else:
                    self.model_type = 'basic'
                    self.IMG_HEIGHT = 224
                    self.IMG_WIDTH = 224
                    print(f"✅ Basic model loaded from {self.model_path} (224x224)")
                    
            elif os.path.exists(self.fallback_model):
                self.model = tf.keras.models.load_model(self.fallback_model, compile=False)
                self.model_type = 'basic'
                self.IMG_HEIGHT = 224
                self.IMG_WIDTH = 224
                print(f"⚠️ Using fallback model from {self.fallback_model}")
            else:
                raise FileNotFoundError("No model file found")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_class_names(self):
        """Load class names from file with fallback support"""
        try:
            # Try advanced class names first
            if os.path.exists(self.class_names_path):
                with open(self.class_names_path, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
            else:
                # Fallback to basic class names
                with open('class_names.txt', 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
            
            print(f"✅ Loaded {len(self.class_names)} class names")
            
        except Exception as e:
            print(f"❌ Error loading class names: {e}")
            raise
    
    def enhance_image(self, image):
        """Apply image enhancement techniques"""
        # Convert to PIL if it's not already
        if not isinstance(image, Image.Image):
            image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Apply subtle enhancements
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.05)
        
        return image
    
    def preprocess_image(self, image_path, enhance=True):
        """
        Advanced image preprocessing with enhancement options
        
        Args:
            image_path (str): Path to the image file
            enhance (bool): Whether to apply image enhancement
            
        Returns:
            np.array: Preprocessed image array
        """
        try:
            # Load and resize image
            image = Image.open(image_path)
            image = image.convert('RGB')
            
            # Apply enhancement if requested
            if enhance:
                image = self.enhance_image(image)
            
            # Resize to model input size
            image = image.resize((self.IMG_WIDTH, self.IMG_HEIGHT), Image.Resampling.LANCZOS)
            
            # Convert to array and normalize
            image_array = np.array(image)
            image_array = image_array.astype('float32') / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            raise
    
    def test_time_augmentation(self, image_array, num_augmentations=5):
        """Apply test-time augmentation for better predictions"""
        predictions = []
        
        # Original image
        pred = self.model.predict(image_array, verbose=0)
        predictions.append(pred)
        
        # Create augmented versions
        original_image = image_array[0]
        
        for _ in range(num_augmentations - 1):
            # Random transformations
            augmented = original_image.copy()
            
            # Random brightness adjustment
            brightness_factor = np.random.uniform(0.9, 1.1)
            augmented = np.clip(augmented * brightness_factor, 0, 1)
            
            # Random rotation (small angles)
            if np.random.random() > 0.5:
                # Simple flip augmentation
                if np.random.random() > 0.5:
                    augmented = np.fliplr(augmented)
            
            # Add noise
            noise = np.random.normal(0, 0.01, augmented.shape)
            augmented = np.clip(augmented + noise, 0, 1)
            
            # Predict on augmented image
            augmented_batch = np.expand_dims(augmented, axis=0)
            pred = self.model.predict(augmented_batch, verbose=0)
            predictions.append(pred)
        
        # Average all predictions
        final_prediction = np.mean(predictions, axis=0)
        return final_prediction
    
    def predict(self, image_path, top_n=5, use_tta=True, enhance_image=True):
        """
        Make advanced prediction on an image
        
        Args:
            image_path (str): Path to the image file
            top_n (int): Number of top predictions to return
            use_tta (bool): Whether to use test-time augmentation
            enhance_image (bool): Whether to enhance the image
            
        Returns:
            dict: Comprehensive prediction results
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path, enhance=enhance_image)
            
            # Make prediction (with or without TTA)
            if use_tta and self.model_type == 'advanced':
                predictions = self.test_time_augmentation(processed_image)
                predictions = predictions[0]  # Remove batch dimension
            else:
                predictions = self.model.predict(processed_image, verbose=0)
                predictions = predictions[0]  # Remove batch dimension
            
            # Get top N predictions
            top_indices = np.argsort(predictions)[-top_n:][::-1]
            
            results = []
            for idx in top_indices:
                class_name = self.class_names[idx]
                confidence = float(predictions[idx])
                results.append((class_name, confidence))
            
            # Format comprehensive results
            formatted_results = self.format_comprehensive_results(results, use_tta, enhance_image)
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            raise
    
    def predict_image_from_array(self, image_array, top_n=5, use_tta=True):
        """
        Make prediction on an image array (for web uploads)
        
        Args:
            image_array (np.array): Image array
            top_n (int): Number of top predictions to return
            use_tta (bool): Whether to use test-time augmentation
            
        Returns:
            dict: Comprehensive prediction results
        """
        try:
            # Ensure proper shape and type
            if len(image_array.shape) == 3:
                image_array = np.expand_dims(image_array, axis=0)
            
            # Normalize if needed
            if image_array.max() > 1.0:
                image_array = image_array.astype('float32') / 255.0
            
            # Resize if needed
            if image_array.shape[1] != self.IMG_HEIGHT or image_array.shape[2] != self.IMG_WIDTH:
                # Convert to PIL Image for resizing
                img = Image.fromarray((image_array[0] * 255).astype(np.uint8))
                img = img.resize((self.IMG_WIDTH, self.IMG_HEIGHT), Image.Resampling.LANCZOS)
                image_array = np.expand_dims(np.array(img).astype('float32') / 255.0, axis=0)
            
            # Make prediction
            if use_tta and self.model_type == 'advanced':
                predictions = self.test_time_augmentation(image_array)
                predictions = predictions[0]
            else:
                predictions = self.model.predict(image_array, verbose=0)
                predictions = predictions[0]
            
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
        
        Args:
            results (list): List of tuples (class_name, confidence)
            used_tta (bool): Whether TTA was used
            enhanced_image (bool): Whether image was enhanced
            
        Returns:
            dict: Comprehensive formatted results
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
        return {
            'model_type': self.model_type,
            'model_path': self.model_path if self.model_type == 'advanced' else self.fallback_model,
            'input_size': f"{self.IMG_WIDTH}x{self.IMG_HEIGHT}",
            'num_classes': len(self.class_names),
            'supports_tta': self.model_type == 'advanced'
        }

# Test function for the advanced predictor
def test_advanced_predictor():
    """Test the advanced predictor"""
    try:
        predictor = AdvancedPlantDiseasePredictor()
        print("✅ Advanced predictor initialized successfully")
        
        model_info = predictor.get_model_info()
        print(f"📊 Model Info: {model_info}")
        
        # Test with sample images if available
        test_dir = 'test'
        if os.path.exists(test_dir):
            test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if test_images:
                test_image = os.path.join(test_dir, test_images[0])
                print(f"\n🧪 Testing with: {test_images[0]}")
                
                results = predictor.predict(test_image, use_tta=True, enhance_image=True)
                
                print(f"🎯 Top prediction: {results['plant']} - {results['disease']}")
                print(f"📈 Confidence: {results['confidence_percentage']} ({results['confidence_level']})")
                print(f"🏥 Severity: {results['disease_info']['severity']}")
                print(f"💊 Treatment: {results['disease_info']['treatment'][:50]}...")
                
        return predictor
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

if __name__ == "__main__":
    test_advanced_predictor()