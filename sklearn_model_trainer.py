"""
Scikit-learn Model Trainer for Plant Disease Classification
This script extracts features from images and trains various sklearn models
"""

import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import cv2
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from feature_extractor import PlantDiseaseFeatureExtractor
warnings.filterwarnings('ignore')


class PlantDiseaseModelTrainer:
    """Train and evaluate scikit-learn models for plant disease classification"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.feature_extractor = PlantDiseaseFeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
    def load_data(self):
        """Load and preprocess the dataset"""
        print("🔄 Loading dataset and extracting features...")
        
        X = []
        y = []
        class_names = []
        
        # Get all class directories
        class_dirs = [d for d in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, d))]
        class_names = sorted(class_dirs)
        
        print(f"📊 Found {len(class_names)} classes")
        
        # Process each class
        for class_idx, class_name in enumerate(tqdm(class_names, desc="Processing classes")):
            class_path = os.path.join(self.data_dir, class_name)
            
            # Get all images in class directory
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"  📁 {class_name}: {len(image_files)} images")
            
            # Process each image (limit to 20 for faster training)
            for img_file in image_files[:20]:  # Limit to 20 images per class for faster training
                img_path = os.path.join(class_path, img_file)
                
                try:
                    # Load image
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                        
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Extract features
                    features = self.feature_extractor.extract_features(image)
                    
                    X.append(features)
                    y.append(class_name)
                    
                except Exception as e:
                    print(f"❌ Error processing {img_path}: {e}")
                    continue
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"📊 Feature shape: {X.shape}")
        print(f"🎯 Classes: {len(class_names)}")
        
        # Save class names
        with open('sklearn_class_names.txt', 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")
        
        return X, y_encoded, class_names
    
    def train_models(self, X, y):
        """Train multiple scikit-learn models"""
        print("\n🧠 Training scikit-learn models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models to train (optimized for speed)
        models_to_train = {
            'RandomForest': RandomForestClassifier(
                n_estimators=50,  # Reduced for speed
                max_depth=10,     # Reduced for speed
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        }
        
        results = {}
        
        # Train each model
        for name, model in models_to_train.items():
            print(f"\n🔄 Training {name}...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            train_score = pipeline.score(X_train, y_train)
            test_score = pipeline.score(X_test, y_test)
            
            # Cross-validation (reduced for speed)
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3)
            
            results[name] = {
                'model': pipeline,
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  📊 Train Accuracy: {train_score:.4f}")
            print(f"  📊 Test Accuracy: {test_score:.4f}")
            print(f"  📊 CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Track best model
            if test_score > self.best_score:
                self.best_score = test_score
                self.best_model = pipeline
                self.best_model_name = name
        
        self.models = results
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        print(f"\n🏆 Best model: {self.best_model_name} (Test Accuracy: {self.best_score:.4f})")
        
        return results
    
    def save_model(self, filename='plant_disease_sklearn_model.pkl'):
        """Save the best model using pickle"""
        print(f"\n💾 Saving best model ({self.best_model_name}) to {filename}...")
        
        model_data = {
            'model': self.best_model,
            'feature_extractor': self.feature_extractor,
            'label_encoder': self.label_encoder,
            'model_name': self.best_model_name,
            'test_accuracy': self.best_score,
            'class_names': self.label_encoder.classes_.tolist()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Also save with joblib for better sklearn compatibility
        joblib.dump(model_data, filename.replace('.pkl', '_joblib.pkl'))
        
        print(f"✅ Model saved successfully!")
        print(f"📊 Model: {self.best_model_name}")
        print(f"🎯 Test Accuracy: {self.best_score:.4f}")
        print(f"📁 Files: {filename} and {filename.replace('.pkl', '_joblib.pkl')}")
        
        return filename
    
    def generate_report(self):
        """Generate a detailed model performance report"""
        print("\n📈 Generating performance report...")
        
        # Predictions on test set
        y_pred = self.best_model.predict(self.X_test)
        
        # Classification report
        report = classification_report(
            self.y_test, y_pred, 
            target_names=[self.label_encoder.classes_[i] for i in range(len(self.label_encoder.classes_))],
            output_dict=True
        )
        
        # Save report
        with open('sklearn_model_report.txt', 'w') as f:
            f.write("Plant Disease Classification - Scikit-learn Model Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Best Model: {self.best_model_name}\n")
            f.write(f"Test Accuracy: {self.best_score:.4f}\n\n")
            
            f.write("Model Comparison:\n")
            f.write("-" * 40 + "\n")
            for name, results in self.models.items():
                f.write(f"{name}:\n")
                f.write(f"  Train Accuracy: {results['train_score']:.4f}\n")
                f.write(f"  Test Accuracy: {results['test_score']:.4f}\n")
                f.write(f"  CV Score: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}\n\n")
            
            f.write("\nDetailed Classification Report:\n")
            f.write("-" * 40 + "\n")
            f.write(classification_report(
                self.y_test, y_pred,
                target_names=[self.label_encoder.classes_[i] for i in range(len(self.label_encoder.classes_))]
            ))
        
        print("✅ Report saved to sklearn_model_report.txt")

def main():
    """Main training function"""
    print("🌱 Plant Disease Classification - Scikit-learn Model Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = PlantDiseaseModelTrainer('train')
    
    # Load data
    X, y, class_names = trainer.load_data()
    
    # Train models
    results = trainer.train_models(X, y)
    
    # Save best model
    model_file = trainer.save_model()
    
    # Generate report
    trainer.generate_report()
    
    print("\n🎉 Training completed successfully!")
    print(f"📁 Model saved as: {model_file}")
    print(f"📊 Best model: {trainer.best_model_name}")
    print(f"🎯 Test accuracy: {trainer.best_score:.4f}")

if __name__ == "__main__":
    main()