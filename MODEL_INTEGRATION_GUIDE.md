# PyTorch Model Integration Guide

## Overview

Your Plant Doc application now has a complete backend to serve your `.pth` PyTorch model. Here's how to integrate your specific model:

## 🎯 Quick Start

### 1. Set Up the Backend

```bash
cd backend
./setup.sh
```

### 2. Add Your Model

1. **Copy your `.pth` file** to the `backend/` directory
2. **Update the model path** in `backend/app.py` if needed:
   ```python
   MODEL_PATH = "your_model_name.pth"  # Change this to your file name
   ```

### 3. Update Class Names

**This is the most important step!** Replace the `CLASS_NAMES` list in `backend/app.py` with your model's actual output classes:

```python
CLASS_NAMES = [
    "Your_Disease_1",
    "Your_Disease_2", 
    "Healthy_Plant",
    # Add all your model's output classes here
]
```

### 4. Start the Backend

```bash
cd backend
python app.py
```

### 5. Test the Integration

Your React frontend will now automatically use your real model instead of mock data!

## 🔧 Model-Specific Configurations

### If Your Model Has Different Architecture

If you get errors loading your model, you may need to modify the `load_model()` function:

```python
def load_model():
    # Option 1: If your model was saved with state_dict only
    model = YourModelClass()  # Define your model architecture here
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

    # Option 2: If your model needs specific device/configuration
    model = torch.load(MODEL_PATH, map_location='cpu')
    # Add any model-specific configuration here
    model.eval()
    return model
```

### If Your Model Uses Different Image Preprocessing

Update the `transform` in `app.py` to match your training preprocessing:

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Change size if needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Update if different
])
```

### If Your Model Outputs Different Format

Modify the `predict_disease()` function if your model outputs probabilities differently:

```python
def predict_disease(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        # Modify this based on your model's output format
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        # ... rest of the function
```

## 📊 Common Model Types

### ResNet-based Models
```python
# Usually works with default configuration
model = torch.load(MODEL_PATH, map_location='cpu')
model.eval()
```

### Custom CNN Models
```python
# You might need to define the architecture first
class YourCustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define your architecture here
        
    def forward(self, x):
        # Define forward pass
        return x

model = YourCustomModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
```

### Transfer Learning Models
```python
# Often saved with the full model
model = torch.load(MODEL_PATH, map_location='cpu')
model.eval()
```

## 🧪 Testing Your Model

### 1. Test the Backend Directly

```bash
curl -X POST -F "image=@path/to/test/image.jpg" http://localhost:8000/api/detect-disease
```

### 2. Check Health Status

```bash
curl http://localhost:8000/health
```

### 3. Test from Frontend

1. Start the backend: `python app.py`
2. Start the frontend: `npm start`
3. Upload an image in the web interface

## 🐛 Troubleshooting

### Model Loading Errors

**Error**: `RuntimeError: Error(s) in loading state_dict`
**Solution**: Your model architecture doesn't match. Define the model architecture before loading:

```python
def load_model():
    # Define your model architecture
    model = YourModelClass()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model
```

### Image Processing Errors

**Error**: `RuntimeError: Expected 4D input`
**Solution**: Check your image preprocessing. Make sure the image tensor has the right dimensions.

### Class Name Mismatch

**Error**: Model predicts but shows wrong disease names
**Solution**: Update the `CLASS_NAMES` list to match your model's training classes exactly.

### Memory Issues

**Error**: `CUDA out of memory` or similar
**Solution**: The backend is configured to use CPU. If you want GPU, modify the `map_location` parameter.

## 🚀 Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "app.py"]
```

## 📝 Example: Complete Integration

Here's a complete example for a typical plant disease model:

```python
# In backend/app.py

# 1. Update model path
MODEL_PATH = "plant_disease_model.pth"

# 2. Update class names (example for common plant diseases)
CLASS_NAMES = [
    "Apple_Scab",
    "Apple_Black_Rot", 
    "Apple_Cedar_Rust",
    "Apple_Healthy",
    "Blueberry_Healthy",
    "Cherry_Powdery_Mildew",
    "Cherry_Healthy",
    "Corn_Common_Rust",
    "Corn_Northern_Leaf_Blight",
    "Corn_Healthy",
    "Grape_Black_Rot",
    "Grape_Esca",
    "Grape_Healthy",
    "Peach_Bacterial_Spot",
    "Peach_Healthy",
    "Pepper_Bacterial_Spot",
    "Pepper_Healthy",
    "Potato_Early_Blight",
    "Potato_Late_Blight",
    "Potato_Healthy",
    "Tomato_Bacterial_Spot",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_Leaf_Spot",
    "Tomato_Spider_Mites",
    "Tomato_Target_Spot",
    "Tomato_Yellow_Leaf_Curl",
    "Tomato_Mosaic_Virus",
    "Tomato_Healthy"
]

# 3. Load model (usually works as-is)
def load_model():
    model = torch.load(MODEL_PATH, map_location='cpu')
    model.eval()
    return model
```

## 🎉 Success!

Once everything is set up:

1. ✅ Your `.pth` model will be loaded and ready
2. ✅ Images will be preprocessed correctly
3. ✅ Predictions will be made using your trained model
4. ✅ Results will be displayed in the beautiful React frontend
5. ✅ Detailed disease information will come from Gemini API

Your Plant Doc application will be fully functional with your custom PyTorch model!
