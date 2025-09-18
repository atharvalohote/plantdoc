#!/bin/bash

# Plant Disease Detection Backend Setup Script

echo "🌱 Setting up Plant Disease Detection Backend..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python version: $PYTHON_VERSION"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3."
    exit 1
fi

# Create virtual environment (optional but recommended)
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if model file exists
if [ ! -f "model.pth" ]; then
    echo "⚠️  Model file 'model.pth' not found!"
    echo "Please place your .pth model file in the backend directory."
    echo "You can also update the MODEL_PATH in app.py to point to your model file."
fi

echo "🎉 Backend setup complete!"
echo ""
echo "Next steps:"
echo "1. Place your .pth model file in the backend directory"
echo "2. Update CLASS_NAMES in app.py to match your model's output classes"
echo "3. Start the backend server: python app.py"
echo "4. The API will be available at http://localhost:8000"
echo ""
echo "To activate the virtual environment in the future:"
echo "source venv/bin/activate"
echo ""
echo "Happy coding! 🌱"
