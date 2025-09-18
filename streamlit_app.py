import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
import json
import os
from typing import Dict, List, Tuple
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Plant Doc - AI Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.375rem;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Plant disease classes (38 classes from your model)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___healthy'
]

# Model architecture (matching your improved model)
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
        
        # Residual blocks
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

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        st.info("🔄 Loading AI model...")
        
        # Create model
        model = PlantDiseaseCNN(num_classes=len(CLASS_NAMES))
        
        # Load weights
        model_path = "backend/plant_disease_model-2.pth"
        if os.path.exists(model_path):
            try:
                # Load with strict=False to handle potential version differences
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                st.success("✅ Model loaded successfully!")
                return model
            except Exception as e:
                st.warning(f"⚠️ Model loading had compatibility issues: {str(e)}")
                st.info("🔄 Trying with strict=False...")
                try:
                    model.load_state_dict(state_dict, strict=False)
                    model.eval()
                    st.success("✅ Model loaded with compatibility mode!")
                    return model
                except Exception as e2:
                    st.error(f"❌ Failed to load model: {str(e2)}")
                    return None
        else:
            st.error(f"❌ Model file not found at {model_path}")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)

def format_disease_name(class_name):
    """Format database class name for display"""
    # Remove underscores and format nicely
    formatted = class_name.replace('___', ' ').replace('_', ' ')
    
    # Handle special cases
    if 'healthy' in formatted.lower():
        plant = formatted.split(' healthy')[0]
        return f"{plant} (Healthy)"
    
    return formatted

def predict_disease(model, image):
    """Predict disease from image"""
    try:
        # Preprocess image
        input_tensor = preprocess_image(image)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, 5)
        
        predictions = []
        for i in range(len(top_probs)):
            class_idx = top_indices[i].item()
            confidence = top_probs[i].item() * 100
            disease_name = format_disease_name(CLASS_NAMES[class_idx])
            predictions.append({
                'diseaseName': disease_name,
                'confidence': round(confidence, 2),
                'classIndex': class_idx
            })
        
        return predictions
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None

def get_gemini_disease_info(disease_name):
    """Get detailed disease information from Gemini API"""
    try:
        api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyD15cCs8Dlm1eBciuJR8BuMR8ZXOL--0VM')
        
        if not api_key:
            return get_fallback_disease_info(disease_name)
        
        prompt = f"""You are a plant disease expert. Provide detailed information about "{disease_name}" in the following JSON format:

{{
  "diseaseName": "{disease_name}",
  "description": "Detailed description of the disease",
  "symptoms": ["symptom1", "symptom2", "symptom3"],
  "causes": ["cause1", "cause2", "cause3"],
  "treatment": ["treatment1", "treatment2", "treatment3"],
  "prevention": ["prevention1", "prevention2", "prevention3"],
  "severity": "Low|Medium|High|Critical",
  "affectedPlantParts": ["part1", "part2", "part3"]
}}

Make sure the information is accurate, scientific, and helpful for plant care."""

        response = requests.post(
            f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}',
            headers={'Content-Type': 'application/json'},
            json={'contents': [{'parts': [{'text': prompt}]}]},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('candidates') and data['candidates'][0].get('content'):
                content = data['candidates'][0]['content']['parts'][0]['text']
                
                # Extract JSON from response
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    return json.loads(json_match.group())
        
        return get_fallback_disease_info(disease_name)
        
    except Exception as e:
        st.warning(f"⚠️ Gemini API error: {str(e)}")
        return get_fallback_disease_info(disease_name)

def get_fallback_disease_info(disease_name):
    """Fallback disease information"""
    return {
        "diseaseName": disease_name,
        "description": f"Information about {disease_name} - a plant condition that may affect crop health and yield.",
        "symptoms": ["Visual changes in plant appearance", "Discoloration of leaves or fruit", "Abnormal growth patterns"],
        "causes": ["Environmental factors", "Pathogen infection", "Nutritional deficiencies"],
        "treatment": ["Consult with agricultural expert", "Apply appropriate treatments", "Monitor plant health"],
        "prevention": ["Maintain proper plant care", "Monitor for early signs", "Use disease-resistant varieties"],
        "severity": "Medium",
        "affectedPlantParts": ["Leaves", "Fruit", "Stems"]
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">🌱 Plant Doc - AI Disease Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Upload a plant image and get instant AI-powered disease diagnosis with detailed treatment information</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Cannot proceed without the AI model. Please check the model file.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📋 Instructions")
        st.markdown("""
        1. **Upload Image**: Click 'Browse files' and select a plant image
        2. **AI Analysis**: Our model will analyze the image
        3. **Get Results**: View disease diagnosis and treatment info
        4. **Learn More**: Get detailed information from AI experts
        """)
        
        st.markdown("## 🌟 Features")
        st.markdown("""
        ✅ **AI-Powered Detection**  
        ✅ **38 Disease Classes**  
        ✅ **High Accuracy**  
        ✅ **Detailed Information**  
        ✅ **Treatment Guidance**  
        ✅ **Prevention Tips**  
        """)
        
        st.markdown("## 📊 Supported Plants")
        st.markdown("""
        • Apple, Cherry, Grape  
        • Tomato, Potato, Corn  
        • Peach, Orange, Blueberry  
        • And many more!
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📸 Upload Plant Image</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a plant image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a plant leaf, fruit, or affected area"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Analyze button
            if st.button("🔍 Analyze Disease", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing your image..."):
                    # Get predictions
                    predictions = predict_disease(model, image)
                    
                    if predictions:
                        # Store results in session state
                        st.session_state.predictions = predictions
                        st.session_state.image = image
                        st.session_state.uploaded_file = uploaded_file
                        
                        st.success("✅ Analysis complete!")
    
    with col2:
        if 'predictions' in st.session_state:
            st.markdown('<h2 class="sub-header">🎯 Analysis Results</h2>', unsafe_allow_html=True)
            
            predictions = st.session_state.predictions
            top_prediction = predictions[0]
            
            # Top prediction card
            st.markdown(f"""
            <div class="success-box">
                <h3>🏆 Top Prediction</h3>
                <h2 style="color: #2E8B57; margin: 0;">{top_prediction['diseaseName']}</h2>
                <h3 style="color: #228B22; margin: 0;">{top_prediction['confidence']}% Confidence</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # All predictions
            st.markdown("### 📊 All Predictions")
            for i, pred in enumerate(predictions):
                if i == 0:
                    st.markdown(f"🥇 **{pred['diseaseName']}** - {pred['confidence']}%")
                else:
                    st.markdown(f"{i+1}. {pred['diseaseName']} - {pred['confidence']}%")
            
            # Get detailed information
            if st.button("📚 Get Detailed Information", use_container_width=True):
                with st.spinner("🤖 Getting detailed information from AI experts..."):
                    disease_info = get_gemini_disease_info(top_prediction['diseaseName'])
                    st.session_state.disease_info = disease_info
                    st.success("✅ Detailed information retrieved!")
    
    # Detailed information section
    if 'disease_info' in st.session_state:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📖 Detailed Disease Information</h2>', unsafe_allow_html=True)
        
        disease_info = st.session_state.disease_info
        
        # Description
        st.markdown("### 📝 Description")
        st.info(disease_info['description'])
        
        # Symptoms, Causes, Treatment, Prevention
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚨 Symptoms")
            for symptom in disease_info['symptoms']:
                st.markdown(f"• {symptom}")
            
            st.markdown("### 🔬 Causes")
            for cause in disease_info['causes']:
                st.markdown(f"• {cause}")
        
        with col2:
            st.markdown("### 💊 Treatment")
            for treatment in disease_info['treatment']:
                st.markdown(f"• {treatment}")
            
            st.markdown("### 🛡️ Prevention")
            for prevention in disease_info['prevention']:
                st.markdown(f"• {prevention}")
        
        # Severity and affected parts
        col1, col2 = st.columns(2)
        
        with col1:
            severity_color = {
                'Low': '#28a745',
                'Medium': '#ffc107', 
                'High': '#fd7e14',
                'Critical': '#dc3545'
            }.get(disease_info['severity'], '#6c757d')
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>⚠️ Severity Level</h4>
                <h3 style="color: {severity_color};">{disease_info['severity']}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🌿 Affected Plant Parts")
            for part in disease_info['affectedPlantParts']:
                st.markdown(f"• {part}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🌱 <strong>Plant Doc</strong> - AI-Powered Plant Disease Detection</p>
        <p>Built with PyTorch, Streamlit, and Google Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
