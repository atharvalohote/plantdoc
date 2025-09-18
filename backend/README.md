# Plant Disease Detection Backend

This is a Flask API server that serves your PyTorch model for plant disease detection.

## Setup Instructions

### 1. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Add Your Model

1. Place your `.pth` model file in the `backend/` directory
2. Update the `MODEL_PATH` variable in `app.py` if your model has a different name
3. Update the `CLASS_NAMES` list in `app.py` to match your model's output classes

### 3. Update Model Architecture (if needed)

If your model has a different architecture, you may need to modify the `load_model()` function in `app.py`. Common modifications:

```python
# For models saved with state_dict only
def load_model():
    model = YourModelClass()  # Define your model architecture
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

# For models with custom architecture
def load_model():
    model = torch.load(MODEL_PATH, map_location='cpu')
    # If you need to modify the model structure
    model.eval()
    return model
```

### 4. Update Class Names

Replace the `CLASS_NAMES` list with your model's actual class names:

```python
CLASS_NAMES = [
    "Your_Disease_1",
    "Your_Disease_2", 
    "Healthy_Plant",
    # ... add all your classes
]
```

### 5. Run the Server

```bash
python app.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### Health Check
```
GET /health
```

### Disease Detection
```
POST /api/detect-disease
Content-Type: multipart/form-data
Body: image file

Response:
{
  "success": true,
  "data": {
    "diseaseName": "Leaf Spot",
    "confidence": 87.5
  },
  "allPredictions": [
    {"diseaseName": "Leaf Spot", "confidence": 87.5},
    {"diseaseName": "Powdery Mildew", "confidence": 8.2},
    {"diseaseName": "Healthy", "confidence": 4.3}
  ]
}
```

### All Predictions (Optional)
```
POST /api/predictions
Content-Type: multipart/form-data
Body: image file

Response:
{
  "success": true,
  "predictions": [
    {"diseaseName": "Disease 1", "confidence": 87.5},
    {"diseaseName": "Disease 2", "confidence": 8.2},
    {"diseaseName": "Disease 3", "confidence": 4.3}
  ]
}
```

## Testing

You can test the API using curl:

```bash
curl -X POST -F "image=@path/to/your/image.jpg" http://localhost:8000/api/detect-disease
```

## Integration with Frontend

Once the backend is running, your React frontend will automatically connect to it. The frontend is configured to call:
- `http://localhost:8000/api/detect-disease` for disease detection
- `http://localhost:8000/health` for health checks

## Troubleshooting

### Model Loading Issues
- Check that your `.pth` file is in the correct location
- Verify the model was saved correctly with `torch.save()`
- Check if you need to specify the model architecture before loading

### Image Processing Issues
- Ensure images are in RGB format
- Check image size and format compatibility
- Verify the preprocessing transforms match your training data

### CORS Issues
- The Flask-CORS is already configured
- If you have issues, check the CORS settings in `app.py`

## Production Deployment

For production, consider:
- Using a production WSGI server like Gunicorn
- Adding authentication/rate limiting
- Implementing proper logging
- Using environment variables for configuration
- Adding input validation and error handling
