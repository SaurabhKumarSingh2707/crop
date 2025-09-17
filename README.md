# Plant Disease Prediction System

A complete AI-powered plant disease detection system built with TensorFlow and Flask. This system can identify diseases in plants from leaf images using deep learning.

## Features

- **AI-Powered Detection**: Uses a MobileNetV2-based CNN model trained on plant disease images
- **Web Interface**: User-friendly Flask web application with drag-and-drop image upload
- **Real-time Predictions**: Instant disease detection with confidence scores
- **Multi-Plant Support**: Supports 38 different plant conditions across various crops
- **Mobile Responsive**: Works on desktop and mobile devices

## Supported Plants & Diseases

The system can detect diseases in the following plants:
- **Apple**: Apple scab, Black rot, Cedar apple rust, Healthy
- **Tomato**: Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Tomato mosaic virus, Yellow leaf curl virus, Healthy
- **Potato**: Early blight, Late blight, Healthy
- **Corn**: Cercospora leaf spot, Common rust, Northern leaf blight, Healthy
- **Grape**: Black rot, Esca (Black measles), Leaf blight, Healthy
- **And many more...**

## Project Structure

```
plant-disease-prediction/
│
├── app.py                          # Flask web application
├── predict.py                      # Prediction module
├── train_simple_model_fixed.py     # Model training script
├── preprocess_data.py              # Data preprocessing
├── best_model.h5                   # Trained model (best weights)
├── plant_disease_model.h5          # Final trained model
├── class_names.txt                 # List of disease classes
├── templates/
│   └── index.html                  # Web interface
├── train/                          # Training dataset
├── test/                           # Test images
└── uploads/                        # Uploaded images (created automatically)
```

## Installation & Setup

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager

### 2. Install Dependencies
```bash
pip install tensorflow flask pillow numpy matplotlib
```

### 3. Train the Model (Optional)
If you want to retrain the model:
```bash
python train_simple_model_fixed.py
```

### 4. Run the Web Application
```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`

## Usage

### Web Interface
1. Open your browser and go to `http://127.0.0.1:5000`
2. Upload an image by clicking the upload area or dragging and dropping
3. Click "Analyze Plant" to get predictions
4. View results with confidence scores and disease information

### Python API
```python
from predict import PlantDiseasePredictor

# Initialize predictor
predictor = PlantDiseasePredictor()

# Make prediction
results = predictor.predict('path/to/plant/image.jpg')
formatted_results = predictor.format_prediction_result(results)

print(f"Disease: {formatted_results['top_prediction']}")
print(f"Confidence: {formatted_results['confidence_percentage']}")
```

## Model Performance

- **Architecture**: MobileNetV2 with custom classification head
- **Training Accuracy**: ~90%
- **Validation Accuracy**: ~73%
- **Input Size**: 224x224 pixels
- **Classes**: 38 plant disease categories

## API Endpoints

### GET /
Main web interface

### POST /predict
Upload and analyze plant image
- **Input**: Multipart form with image file
- **Output**: JSON with prediction results

### GET /health
Health check endpoint
- **Output**: System status and model information

## File Formats Supported

- PNG
- JPG/JPEG
- GIF

Maximum file size: 16MB

## Technical Details

### Model Architecture
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Custom Layers**: Global Average Pooling + Dense layers with dropout
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Categorical Crossentropy

### Data Augmentation
- Rotation: ±20 degrees
- Width/Height shift: ±20%
- Shear transformation: ±20%
- Zoom: ±20%
- Horizontal flip
- Normalization: Pixel values scaled to [0,1]

### Training Configuration
- **Batch Size**: 16
- **Epochs**: 30
- **Validation Split**: 20%
- **Callbacks**: Early stopping, Learning rate reduction, Model checkpointing

## Development Notes

### Key Files
- `app.py`: Main Flask application with image upload and prediction endpoints
- `predict.py`: PlantDiseasePredictor class with model loading and prediction logic
- `train_simple_model_fixed.py`: Improved training script with error handling
- `templates/index.html`: Modern, responsive web interface

### Deployment Considerations
- For production, use a WSGI server like Gunicorn
- Consider adding authentication for sensitive deployments
- Implement rate limiting for API endpoints
- Add logging and monitoring

## Troubleshooting

### Common Issues
1. **Model loading errors**: Ensure `best_model.h5` exists in the project directory
2. **Memory errors**: Reduce batch size or image resolution
3. **Slow predictions**: Consider using GPU acceleration or model optimization

### Performance Tips
- Use GPU acceleration for faster training and inference
- Implement model caching to reduce startup time
- Consider using TensorFlow Lite for mobile deployment

## Future Enhancements

- [ ] Add more plant species and diseases
- [ ] Implement treatment recommendations
- [ ] Add user authentication and history
- [ ] Create mobile app version
- [ ] Add batch processing capabilities
- [ ] Implement model versioning and A/B testing

## License

This project is created for educational purposes. Please ensure you have appropriate licenses for any datasets used.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For questions or issues, please check the troubleshooting section or create an issue in the project repository.