# Plant Doc 🌱

An AI-powered plant disease detection and information application built with React and TypeScript.

## 👥 Team Members

- **Atharva Lohote** - Project Lead & Full-Stack Development
- **Indrajeet Chavan** - AI/ML Model Development & Backend
- **Vedant Telgar** - Frontend Development & UI/UX Design

## Features

- **Image Upload**: Drag-and-drop or click to upload plant images
- **AI Disease Detection**: Uses your custom model to detect plant diseases
- **Detailed Information**: Integrates with Gemini API to provide comprehensive disease information
- **Modern UI**: Beautiful, responsive design with Tailwind CSS
- **Real-time Analysis**: Live feedback during the analysis process

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- Your plant disease detection model API
- Gemini API key (optional, falls back to mock data)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Plantdoc
```

2. Install dependencies:
```bash
npm install
```

3. Set up environment variables:
```bash
cp env.example .env
```

Edit `.env` file with your configuration:
```env
REACT_APP_API_BASE_URL=http://localhost:8000
REACT_APP_GEMINI_API_KEY=your_gemini_api_key_here
```

4. Start the development server:
```bash
npm start
```

The application will open at `http://localhost:3000`.

## API Integration

### Disease Detection Model

The app expects your model API to be available at:
```
POST /api/detect-disease
```

**Request**: Form data with image file
**Response**:
```json
{
  "success": true,
  "data": {
    "diseaseName": "Leaf Spot",
    "confidence": 85.5
  }
}
```

### Gemini API Integration

The app uses Google's Gemini API to provide detailed disease information. If the API key is not configured, it falls back to mock data.

## Project Structure

```
src/
├── components/          # React components
│   ├── ImageUpload.tsx  # Image upload with drag-and-drop
│   ├── DiseaseDetection.tsx # AI analysis component
│   └── DiseaseInfo.tsx  # Disease information display
├── services/           # API services
│   └── api.ts         # API integration logic
├── types.ts           # TypeScript type definitions
├── App.tsx            # Main application component
└── index.tsx          # Application entry point
```

## Customization

### Styling
The app uses Tailwind CSS with a custom color scheme. You can modify colors in `tailwind.config.js`:

```javascript
colors: {
  primary: {
    // Your primary color palette
  },
  secondary: {
    // Your secondary color palette
  }
}
```

### API Endpoints
Update the API base URL in `src/services/api.ts` or set the `REACT_APP_API_BASE_URL` environment variable.

## Deployment

### Build for Production
```bash
npm run build
```

### Deploy to Netlify/Vercel
1. Connect your repository
2. Set environment variables in the deployment platform
3. Deploy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support, please open an issue in the repository or contact the development team.
