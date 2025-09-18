import React, { useState } from 'react';
import { Leaf, Camera, Brain, Info, Sparkles, Shield, Zap, Users } from 'lucide-react';
import ImageUpload from './components/ImageUpload';
import DiseaseDetection from './components/DiseaseDetection';
import DiseaseInfo from './components/DiseaseInfo';
import { DiseaseResult } from './types';

function App() {
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [diseaseResult, setDiseaseResult] = useState<DiseaseResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleImageUpload = (file: File) => {
    setUploadedImage(file);
    setDiseaseResult(null);
  };

  const handleDiseaseDetected = (result: DiseaseResult) => {
    setDiseaseResult(result);
  };

  const handleReset = () => {
    setUploadedImage(null);
    setDiseaseResult(null);
    setIsAnalyzing(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 to-secondary-50">
      {/* Enhanced Header */}
      <header className="bg-gradient-to-r from-primary-600 to-secondary-600 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-20">
            <div className="flex items-center space-x-4">
              <div className="flex items-center justify-center w-12 h-12 bg-white/20 backdrop-blur-sm rounded-xl">
                <Leaf className="w-7 h-7 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white flex items-center gap-2">
                  Plant Doc
                  <Sparkles className="w-5 h-5 text-yellow-300" />
                </h1>
                <p className="text-primary-100 text-sm">AI-Powered Plant Disease Detection & Treatment</p>
              </div>
            </div>
            <div className="flex items-center space-x-6">
              <div className="hidden sm:flex items-center space-x-4 text-sm text-white/90">
                <div className="flex items-center space-x-2 bg-white/10 px-3 py-2 rounded-lg">
                  <Camera className="w-4 h-4" />
                  <span>Upload</span>
                </div>
                <div className="flex items-center space-x-2 bg-white/10 px-3 py-2 rounded-lg">
                  <Brain className="w-4 h-4" />
                  <span>Analyze</span>
                </div>
                <div className="flex items-center space-x-2 bg-white/10 px-3 py-2 rounded-lg">
                  <Info className="w-4 h-4" />
                  <span>Learn</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Features Section */}
      <section className="bg-white/50 backdrop-blur-sm py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Why Choose Plant Doc?</h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Advanced AI technology meets plant care expertise to provide accurate disease detection and treatment recommendations.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center p-6 bg-white/70 rounded-xl shadow-sm border border-gray-200">
              <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Brain className="w-8 h-8 text-primary-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">AI-Powered Detection</h3>
              <p className="text-gray-600">Our advanced PyTorch model accurately identifies 38+ plant diseases with high confidence.</p>
            </div>
            
            <div className="text-center p-6 bg-white/70 rounded-xl shadow-sm border border-gray-200">
              <div className="w-16 h-16 bg-secondary-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Zap className="w-8 h-8 text-secondary-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Instant Analysis</h3>
              <p className="text-gray-600">Get real-time disease detection and detailed treatment recommendations in seconds.</p>
            </div>
            
            <div className="text-center p-6 bg-white/70 rounded-xl shadow-sm border border-gray-200">
              <div className="w-16 h-16 bg-accent-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Shield className="w-8 h-8 text-accent-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Expert Knowledge</h3>
              <p className="text-gray-600">Powered by Gemini AI for comprehensive disease information and prevention strategies.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Image Upload and Detection */}
          <div className="space-y-6">
            <ImageUpload 
              onImageUpload={handleImageUpload}
              uploadedImage={uploadedImage}
              onReset={handleReset}
            />
            
            {uploadedImage && (
              <DiseaseDetection
                image={uploadedImage}
                onDiseaseDetected={handleDiseaseDetected}
                isAnalyzing={isAnalyzing}
                setIsAnalyzing={setIsAnalyzing}
              />
            )}
          </div>

          {/* Right Column - Disease Information */}
          <div className="space-y-6">
            {diseaseResult ? (
              <DiseaseInfo diseaseResult={diseaseResult} />
            ) : (
              <div className="card">
                <div className="text-center py-12">
                  <Info className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    Disease Information
                  </h3>
                  <p className="text-gray-500">
                    Upload a plant image to get detailed disease information and treatment recommendations.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Enhanced Meet Our Team Section */}
        <div className="mt-16 mb-8">
          <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-xl border border-gray-200 p-8">
            <div className="text-center mb-8">
              <div className="flex items-center justify-center mb-4">
                <Users className="w-8 h-8 text-primary-600 mr-3" />
                <h2 className="text-3xl font-bold text-gray-900">Meet Our Team</h2>
              </div>
              <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                We're passionate developers working together to revolutionize plant health diagnostics using cutting-edge AI technology.
              </p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {/* Atharva Lohote */}
              <div className="text-center p-6 bg-white/50 rounded-xl shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                <div className="w-24 h-24 bg-gradient-to-br from-primary-100 to-primary-200 rounded-full flex items-center justify-center mb-4 mx-auto shadow-lg">
                  <span className="text-primary-600 font-bold text-2xl">AL</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Atharva Lohote</h3>
                <p className="text-primary-600 font-medium mb-3">Project Lead & Full-Stack Development</p>
                <p className="text-gray-600 text-sm">
                  Leading the development of Plant Doc with expertise in React, TypeScript, and system architecture.
                </p>
              </div>

              {/* Indrajeet Chavan */}
              <div className="text-center p-6 bg-white/50 rounded-xl shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                <div className="w-24 h-24 bg-gradient-to-br from-secondary-100 to-secondary-200 rounded-full flex items-center justify-center mb-4 mx-auto shadow-lg">
                  <span className="text-secondary-600 font-bold text-2xl">IC</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Indrajeet Chavan</h3>
                <p className="text-secondary-600 font-medium mb-3">AI/ML Model Development & Backend</p>
                <p className="text-gray-600 text-sm">
                  Specializing in PyTorch model development, Flask backend, and machine learning optimization.
                </p>
              </div>

              {/* Vedant Telgar */}
              <div className="text-center p-6 bg-white/50 rounded-xl shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                <div className="w-24 h-24 bg-gradient-to-br from-accent-100 to-accent-200 rounded-full flex items-center justify-center mb-4 mx-auto shadow-lg">
                  <span className="text-accent-600 font-bold text-2xl">VT</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Vedant Telgar</h3>
                <p className="text-accent-600 font-medium mb-3">Frontend Development & UI/UX Design</p>
                <p className="text-gray-600 text-sm">
                  Creating beautiful, responsive user interfaces with modern design principles and user experience focus.
                </p>
              </div>
            </div>

            <div className="mt-8 pt-6 border-t border-gray-200 text-center">
              <p className="text-gray-500 italic">
                "Building the future of plant health diagnostics with AI - one leaf at a time."
              </p>
            </div>
          </div>
        </div>
      </main>

      {/* Enhanced Footer */}
      <footer className="bg-gradient-to-r from-gray-900 to-gray-800 text-white mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center md:text-left">
              <div className="flex items-center justify-center md:justify-start mb-4">
                <Leaf className="w-6 h-6 text-primary-400 mr-2" />
                <span className="text-xl font-bold">Plant Doc</span>
              </div>
              <p className="text-gray-300 text-sm">
                AI-powered plant disease detection and treatment recommendations for healthier plants.
              </p>
            </div>
            
            <div className="text-center">
              <h3 className="text-lg font-semibold mb-4">Our Team</h3>
              <div className="space-y-2">
                <div className="text-primary-400 text-sm font-medium">Atharva Lohote</div>
                <div className="text-secondary-400 text-sm font-medium">Indrajeet Chavan</div>
                <div className="text-accent-400 text-sm font-medium">Vedant Telgar</div>
              </div>
            </div>
            
            <div className="text-center md:text-right">
              <h3 className="text-lg font-semibold mb-4">Technology</h3>
              <div className="space-y-1 text-sm text-gray-300">
                <div>PyTorch AI Model</div>
                <div>React & TypeScript</div>
                <div>Gemini AI Integration</div>
                <div>Flask Backend</div>
              </div>
            </div>
          </div>
          
          <div className="border-t border-gray-700 mt-8 pt-8 text-center">
            <p className="text-gray-400 text-sm">
              &copy; 2024 Plant Doc. Built with ❤️ for plant lovers everywhere.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
