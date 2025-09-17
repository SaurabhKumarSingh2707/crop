"""
Simple prediction script for the scikit-learn plant disease model
"""

import pickle
import cv2
import numpy as np
import os

def load_model(model_path='plant_disease_sklearn_model.pkl'):
    """Load the trained scikit-learn model"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_image(image_path, model_data):
    """Predict plant disease from image path"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None, "Could not load image"
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract features
        feature_extractor = model_data['feature_extractor']
        features = feature_extractor.extract_features(image)
        features = features.reshape(1, -1)
        
        # Make prediction
        model = model_data['model']
        probabilities = model.predict_proba(features)[0]
        predicted_class_idx = model.predict(features)[0]
        
        # Get class name
        class_names = model_data['class_names']
        predicted_class = class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'percentage': confidence * 100
        }, None
        
    except Exception as e:
        return None, str(e)

def main():
    """Test the model with a sample prediction"""
    print("🌱 Scikit-learn Plant Disease Predictor")
    print("=" * 40)
    
    # Load model
    model_data = load_model()
    if model_data is None:
        print("❌ Failed to load model")
        return
    
    print(f"✅ Model loaded: {model_data['model_name']}")
    print(f"🎯 Accuracy: {model_data['test_accuracy']:.4f}")
    print(f"📊 Classes: {len(model_data['class_names'])}")
    
    # Check for sample images
    sample_dirs = ['uploads', 'test']
    sample_image = None
    
    for dir_name in sample_dirs:
        if os.path.exists(dir_name):
            image_files = [f for f in os.listdir(dir_name) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                sample_image = os.path.join(dir_name, image_files[0])
                break
    
    if sample_image:
        print(f"\n🔍 Testing with sample image: {sample_image}")
        result, error = predict_image(sample_image, model_data)
        
        if result:
            print(f"🎯 Prediction: {result['class']}")
            print(f"📊 Confidence: {result['percentage']:.2f}%")
        else:
            print(f"❌ Error: {error}")
    else:
        print("\n📝 No sample images found in uploads/ or test/ directories")
        print("   Place an image file in one of these directories to test")

if __name__ == "__main__":
    main()