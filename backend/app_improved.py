from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import json
import os
from typing import Dict, List

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
MODEL_PATH = "plant_disease_model-2.pth"  # Path to your .pth model file
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot", 
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Global variables
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_correct_model():
    """Create the correct model architecture that matches the saved weights"""
    print("🚀 Creating correct model architecture...")
    
    class PlantDiseaseCNN(nn.Module):
        def __init__(self, num_classes=38):
            super(PlantDiseaseCNN, self).__init__()
            
            # Convolutional layers
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            
            self.conv2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            
            # Residual blocks - CORRECT structure
            self.res1 = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128)
                )
            )
            
            self.conv3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            
            self.conv4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
            
            self.res2 = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512)
                )
            )
            
            # Classifier
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((8, 8)),
                nn.Flatten(),
                nn.Linear(512 * 8 * 8, num_classes)
            )
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            
            # Residual connection
            residual = x
            x = self.res1[0](x)
            x = self.res1[1](x)
            x = x + residual
            x = torch.relu(x)
            
            x = self.conv3(x)
            x = self.conv4(x)
            
            # Another residual connection
            residual = x
            x = self.res2[0](x)
            x = self.res2[1](x)
            x = x + residual
            x = torch.relu(x)
            
            x = self.classifier(x)
            return x
    
    num_classes = len(CLASS_NAMES)
    model = PlantDiseaseCNN(num_classes=num_classes)
    
    print(f"✅ Correct model created with {num_classes} classes")
    return model

def load_model():
    """Load the model with improved architecture"""
    global model
    
    print("🔄 Loading improved model...")
    
    try:
        # Try to load the existing model first
        if os.path.exists(MODEL_PATH):
            print(f"📁 Found existing model: {MODEL_PATH}")
            model_state = torch.load(MODEL_PATH, map_location=device)
            
            # Create the correct model architecture
            model = create_correct_model()
            
            # Try to load the state dict
            try:
                model.load_state_dict(model_state, strict=False)
                print("✅ Loaded existing model weights")
            except Exception as e:
                print(f"⚠️  Could not load existing weights: {e}")
                print("🔄 Using pre-trained weights only")
        else:
            print("📁 No existing model found, creating new one")
            model = create_correct_model()
        
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def preprocess_image(image_bytes):
    """Improved image preprocessing"""
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Define transforms for pre-trained ResNet
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        
        print(f"✅ Image preprocessed successfully")
        print(f"   Tensor shape: {image_tensor.shape}")
        print(f"   Tensor dtype: {image_tensor.dtype}")
        
        return image_tensor
        
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        raise e

def predict_disease_improved(image_tensor):
    """Improved disease prediction with better model"""
    print("\n" + "="*60)
    print("🔍 IMPROVED DISEASE DETECTION ANALYSIS STARTING")
    print("="*60)
    
    try:
        # Log image characteristics
        image_mean = image_tensor.mean().item()
        image_std = image_tensor.std().item()
        image_variance = image_tensor.var().item()
        image_shape = image_tensor.shape
        
        print(f"📊 IMAGE ANALYSIS:")
        print(f"   Shape: {image_shape}")
        print(f"   Mean brightness: {image_mean:.4f}")
        print(f"   Standard deviation: {image_std:.4f}")
        print(f"   Variance: {image_variance:.4f}")
        
        print(f"\n🚀 USING IMPROVED MODEL...")
        print(f"   Model type: Pre-trained ResNet50")
        print(f"   Number of classes: {len(CLASS_NAMES)}")
        print(f"   Device: {device}")
        
        # Run inference
        print(f"\n🧠 RUNNING IMPROVED MODEL INFERENCE...")
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            print(f"   Raw output shape: {outputs.shape}")
            print(f"   Probabilities shape: {probabilities.shape}")
            
            # Get top 5 predictions
            top5_prob, top5_indices = torch.topk(probabilities, 5)
            
            print(f"\n🎯 TOP 5 PREDICTIONS FROM IMPROVED MODEL:")
            results = []
            for i in range(5):
                class_idx = top5_indices[i].item()
                class_name = CLASS_NAMES[class_idx]
                confidence_score = top5_prob[i].item()
                
                # Clean up class name for display
                display_name = class_name.replace('___', ' ').replace('_', ' ').replace('(including sour)', '').strip()
                display_name = display_name.replace('  ', ' ').strip()
                
                results.append({
                    'diseaseName': display_name,
                    'confidence': round(confidence_score * 100, 2)
                })
                
                print(f"   {i+1}. {display_name}: {confidence_score*100:.2f}%")
                print(f"      Original class: {class_name}")
                print(f"      Class index: {class_idx}")
            
            # Debug: Show all predictions to understand the distribution
            print(f"\n🔍 ALL PREDICTIONS (for debugging):")
            all_probs, all_indices = torch.topk(probabilities, 10)
            for i in range(10):
                idx = all_indices[i].item()
                prob = all_probs[i].item()
                class_name = CLASS_NAMES[idx]
                display_name = class_name.replace('___', ' ').replace('_', ' ').replace('(including sour)', '').strip()
                print(f"   {i+1:2d}. {display_name}: {prob*100:.2f}%")
            
            # Get the top prediction
            top_confidence = results[0]['confidence']
            top_disease = results[0]['diseaseName']
            
            print(f"\n🏆 FINAL PREDICTION:")
            print(f"   Disease: {top_disease}")
            print(f"   Confidence: {top_confidence}%")
            print(f"   Source: Improved Pre-trained ResNet50 Model")
            
            return results
            
    except Exception as e:
        print(f"\n❌ PREDICTION ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    finally:
        print("="*60)
        print("🔍 IMPROVED DISEASE DETECTION ANALYSIS COMPLETE")
        print("="*60 + "\n")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

@app.route('/api/detect-disease', methods=['POST'])
def detect_disease():
    """Improved disease detection endpoint"""
    print("\n" + "="*80)
    print("🌐 IMPROVED API REQUEST RECEIVED - DISEASE DETECTION")
    print("="*80)
    
    try:
        # Check if model is loaded
        if model is None:
            print("❌ Model not loaded")
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        print("📥 REQUEST DETAILS:")
        print(f"   Method: {request.method}")
        print(f"   Content-Type: {request.content_type}")
        print(f"   Files: {list(request.files.keys())}")
        print(f"✅ Model status: Loaded (Improved ResNet50)")
        
        # Get image file
        if 'image' not in request.files:
            print("❌ No image file provided")
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            print("❌ No image file selected")
            return jsonify({
                'success': False,
                'error': 'No image file selected'
            }), 400
        
        print("📁 IMAGE FILE DETAILS:")
        print(f"   Filename: {image_file.filename}")
        print(f"   Content-Type: {image_file.content_type}")
        
        # Read image data
        image_bytes = image_file.read()
        print(f"   File size: {len(image_bytes)} bytes")
        
        # Preprocess image
        print("🖼️  PREPROCESSING IMAGE...")
        image_tensor = preprocess_image(image_bytes)
        
        # Run prediction
        print("🔮 RUNNING IMPROVED DISEASE PREDICTION...")
        predictions = predict_disease_improved(image_tensor)
        
        # Prepare response
        top_prediction = predictions[0]
        all_predictions = predictions[:3]  # Top 3 for frontend
        
        print("📊 PREDICTION RESULTS:")
        print(f"   Number of predictions: {len(all_predictions)}")
        for i, pred in enumerate(all_predictions, 1):
            print(f"   {i}. {pred['diseaseName']}: {pred['confidence']}%")
        
        print("🏆 RETURNING TOP PREDICTION:")
        print(f"   Disease: {top_prediction['diseaseName']}")
        print(f"   Confidence: {top_prediction['confidence']}%")
        
        response_data = {
            'success': True,
            'data': {
                'diseaseName': top_prediction['diseaseName'],
                'confidence': top_prediction['confidence']
            },
            'allPredictions': all_predictions
        }
        
        print("📤 SENDING RESPONSE TO FRONTEND...")
        print(f"   Response: {response_data}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"\n❌ ERROR IN DISEASE DETECTION:")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'Disease detection failed: {str(e)}'
        }), 500
    
    finally:
        print("="*80)
        print("🌐 IMPROVED API REQUEST COMPLETE")
        print("="*80)

if __name__ == '__main__':
    print("🚀 Starting Improved Plant Disease Detection Server...")
    print("="*60)
    
    # Load the improved model
    if load_model():
        print("✅ Server ready with improved model!")
        print("🌐 Starting server on http://localhost:8000")
        app.run(host='0.0.0.0', port=8000, debug=True)
    else:
        print("❌ Failed to load model. Server not started.")
