"""
Feature extraction module for plant disease classification
This module contains the feature extraction classes used in both training and prediction
"""

import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from sklearn.preprocessing import StandardScaler

class PlantDiseaseFeatureExtractor:
    """Extract multiple types of features from plant disease images"""
    
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        self.scaler = StandardScaler()
        
    def extract_color_histogram(self, image):
        """Extract color histogram features"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calculate histograms for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [50], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [50], [0, 256])
        
        # Normalize and flatten
        hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
        hist_s = hist_s.flatten() / (hist_s.sum() + 1e-7)
        hist_v = hist_v.flatten() / (hist_v.sum() + 1e-7)
        
        return np.concatenate([hist_h, hist_s, hist_v])
    
    def extract_hog_features(self, image):
        """Extract HOG (Histogram of Oriented Gradients) features"""
        gray = rgb2gray(image)
        
        # Extract HOG features (simplified for speed)
        hog_features = hog(
            gray,
            orientations=6,        # Reduced from 9
            pixels_per_cell=(16, 16),  # Increased from (8,8) for speed
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            feature_vector=True
        )
        
        return hog_features
    
    def extract_lbp_features(self, image):
        """Extract Local Binary Pattern features"""
        gray = rgb2gray(image)
        gray = (gray * 255).astype(np.uint8)
        
        # Calculate LBP
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Calculate histogram
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        # Normalize
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        
        return hist
    
    def extract_texture_features(self, image):
        """Extract basic texture features"""
        gray = rgb2gray(image)
        
        # Calculate texture statistics
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mean = np.mean(grad_mag)
        grad_std = np.std(grad_mag)
        
        return np.array([mean_val, std_val, grad_mean, grad_std])
    
    def extract_features(self, image):
        """Extract all features from an image"""
        # Resize image
        image = cv2.resize(image, self.img_size)
        
        # Extract different types of features
        color_features = self.extract_color_histogram(image)
        hog_features = self.extract_hog_features(image)
        lbp_features = self.extract_lbp_features(image)
        texture_features = self.extract_texture_features(image)
        
        # Combine all features
        all_features = np.concatenate([
            color_features,
            hog_features,
            lbp_features,
            texture_features
        ])
        
        return all_features