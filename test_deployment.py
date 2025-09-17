#!/usr/bin/env python3
"""
Test script to verify model loading and prediction functionality
Run this before deploying to ensure everything works correctly
"""

import os
import sys
import pickle
import numpy as np
from feature_extractor import PlantDiseaseFeatureExtractor

def test_model_loading():
    """Test if model files can be loaded"""
    print("🧪 Testing model loading...")
    
    model_files = [
        'plant_disease_sklearn_model.pkl',
        'plant_disease_sklearn_model_joblib.pkl'
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                print(f"📁 Testing {model_file}...")
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Check required components
                required_keys = ['model', 'feature_extractor', 'label_encoder', 'class_names']
                for key in required_keys:
                    if key not in model_data:
                        print(f"❌ Missing key: {key}")
                        return False
                    else:
                        print(f"✅ Found {key}: {type(model_data[key])}")
                
                # Test feature extraction
                print("🔧 Testing feature extraction...")
                feature_extractor = model_data['feature_extractor']
                
                # Create dummy image
                dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                features = feature_extractor.extract_features(dummy_image)
                print(f"✅ Feature extraction successful: {features.shape}")
                
                # Test model prediction
                print("🎯 Testing model prediction...")
                model = model_data['model']
                prediction = model.predict(features.reshape(1, -1))
                probabilities = model.predict_proba(features.reshape(1, -1))
                
                print(f"✅ Prediction successful: class {prediction[0]}")
                print(f"✅ Probabilities shape: {probabilities.shape}")
                print(f"✅ Total classes: {len(model_data['class_names'])}")
                
                return True
                
            except Exception as e:
                print(f"❌ Error loading {model_file}: {e}")
                continue
    
    return False

def test_dependencies():
    """Test if all required dependencies are available"""
    print("📦 Testing dependencies...")
    
    required_packages = [
        'flask', 'numpy', 'sklearn', 'cv2', 'skimage', 
        'PIL', 'pickle', 'joblib'
    ]
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'sklearn':
                import sklearn
            elif package == 'skimage':
                import skimage
            elif package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            return False
    
    return True

def test_file_structure():
    """Test if all required files are present"""
    print("📂 Testing file structure...")
    
    required_files = [
        'app_sklearn.py',
        'feature_extractor.py',
        'requirements.txt',
        'templates/index_sklearn.html'
    ]
    
    optional_files = [
        'plant_disease_sklearn_model.pkl',
        'plant_disease_sklearn_model_joblib.pkl',
        'Procfile',
        'render.yaml'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (REQUIRED)")
            return False
    
    for file_path in optional_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"⚠️ {file_path} (optional)")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Plant Disease Classifier - Deployment Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Dependencies", test_dependencies),
        ("Model Loading", test_model_loading)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if not test_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Ready for deployment.")
        return 0
    else:
        print("❌ Some tests failed. Please fix issues before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())