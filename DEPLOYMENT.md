# Deployment Guide for Plant Disease Classifier

## Option 1: Direct Upload to Render.com

### Steps:
1. **Connect your GitHub repository to Render.com**
2. **Create a new Web Service** 
3. **Configure the service:**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app_sklearn:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
   - Environment: Python 3.9+

### Files needed:
- ✅ `app_sklearn.py` (main Flask app)
- ✅ `feature_extractor.py` (feature extraction module)
- ✅ `requirements.txt` (dependencies)
- ✅ `Procfile` (Render.com configuration)
- ✅ `render.yaml` (optional: service configuration)
- ✅ `plant_disease_sklearn_model.pkl` (model file - 1.3MB)
- ✅ `templates/index_sklearn.html` (web interface)

## Option 2: Using Git LFS for Large Files

If model files are too large for regular Git:

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.pkl"
git add .gitattributes

# Add and commit files
git add .
git commit -m "Add model files with LFS"
git push
```

## Option 3: Download Model from URL

The app is configured to download the model from a URL if not found locally.
Update the `download_model_from_url()` function with your model URL.

## Troubleshooting Common Issues:

### 1. "Model not loaded" error
- **Cause**: Model files not uploaded or corrupted
- **Solution**: Check file sizes, use Git LFS, or enable model download

### 2. Import errors
- **Cause**: Missing dependencies in requirements.txt
- **Solution**: Update requirements.txt with exact versions

### 3. Memory issues
- **Cause**: Large model size
- **Solution**: Use smaller model or increase server memory

### 4. Timeout errors
- **Cause**: Model loading takes too long
- **Solution**: Increase timeout in Procfile to 120 seconds

## Environment Variables (Optional):
- `PORT`: Server port (automatically set by Render.com)
- `MODEL_URL`: URL to download model if not found locally

## Render.com Specific Notes:
- Free tier has 512MB RAM limit
- Build time limit: 15 minutes
- Request timeout: 30 seconds (increased to 120 in our config)
- Files over 100MB should use Git LFS

## Testing Deployment:
1. Test locally: `python app_sklearn.py`
2. Test with gunicorn: `gunicorn app_sklearn:app --bind 0.0.0.0:5002`
3. Check logs for any errors
4. Verify model loading in deployment logs