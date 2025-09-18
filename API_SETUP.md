# API Setup Guide

## Current Status: Demo Mode ✅

Your Plant Doc app is currently running in **Demo Mode** with mock data. This allows you to test the full user experience without needing your model API running.

## What's Working Now

- ✅ Image upload and preview
- ✅ Mock disease detection (random diseases with realistic confidence scores)
- ✅ Gemini API integration for detailed disease information
- ✅ Beautiful UI with all features functional

## To Connect Your Real Model

### 1. Set up your model API

Your model should be accessible at:
```
POST http://localhost:8000/api/detect-disease
```

### 2. Expected API Response Format

Your API should accept:
- **Content-Type**: `multipart/form-data`
- **Body**: Image file in a field named `image`

And return:
```json
{
  "success": true,
  "data": {
    "diseaseName": "Leaf Spot",
    "confidence": 87.5
  }
}
```

### 3. Update Environment Variables

Edit your `.env` file:
```env
REACT_APP_API_BASE_URL=http://your-model-server:port
REACT_APP_GEMINI_API_KEY=AIzaSyD15cCs8Dlm1eBciuJR8BuMR8ZXOL--0VM
```

### 4. Test the Connection

1. Start your model API server
2. Restart the React app: `npm start`
3. Upload an image - it should now use your real model instead of mock data

## Demo Mode Features

While in demo mode, the app will:
- Show a yellow "Demo Mode" notification
- Use random disease names from: Leaf Spot, Powdery Mildew, Root Rot, Bacterial Blight, Rust
- Provide realistic confidence scores (75-95%)
- Still use the real Gemini API for detailed disease information

## Troubleshooting

### If you get 404 errors:
- Check that your model API is running
- Verify the API endpoint URL in `.env`
- Ensure your API accepts POST requests with image data

### If you get CORS errors:
- Add CORS headers to your model API
- Or use a proxy in your React app

### If Gemini API fails:
- The app will fall back to mock disease information
- Check your API key is correct in `.env`

## Next Steps

1. **Test the demo**: Upload some plant images and see the full workflow
2. **Set up your model**: Follow the API setup guide above
3. **Deploy**: When ready, deploy both your model API and this frontend

The app is fully functional and ready for production use once your model API is connected!
