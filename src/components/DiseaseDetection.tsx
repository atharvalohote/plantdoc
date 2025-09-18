import React, { useState } from "react";
import { Brain, Loader2, AlertCircle } from "lucide-react";
import type { DiseaseResult } from "../types";
import { detectDisease, getDiseaseInfo } from "../services/api";

interface DiseaseDetectionProps {
  image: File;
  onDiseaseDetected: (result: DiseaseResult) => void;
  isAnalyzing: boolean;
  setIsAnalyzing: (analyzing: boolean) => void;
}

const DiseaseDetection: React.FC<DiseaseDetectionProps> = ({
  image,
  onDiseaseDetected,
  isAnalyzing,
  setIsAnalyzing
}) => {
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    console.log("\n" + "=".repeat(80));
    console.log("🔍 FRONTEND: DISEASE DETECTION STARTING");
    console.log("=".repeat(80));
    
    if (!image) {
      console.log("❌ ERROR: No image provided");
      return;
    }

    console.log("📁 IMAGE DETAILS:");
    console.log(`   Name: ${image.name}`);
    console.log(`   Size: ${image.size} bytes`);
    console.log(`   Type: ${image.type}`);
    console.log(`   Last Modified: ${new Date(image.lastModified).toISOString()}`);

    setIsAnalyzing(true);
    setError(null);

    try {
      console.log("\n🚀 STEP 1: CALLING DISEASE DETECTION API...");
      console.log("   API Endpoint: /api/detect-disease");
      console.log("   Method: POST");
      console.log("   Payload: FormData with image file");
      
      // Step 1: Detect disease using your model
      const detectionResult = await detectDisease(image);
      
      console.log("📊 DISEASE DETECTION API RESPONSE:");
      console.log(`   Success: ${detectionResult.success}`);
      if (detectionResult.success && detectionResult.data) {
        console.log(`   Disease: ${detectionResult.data.diseaseName}`);
        console.log(`   Confidence: ${detectionResult.data.confidence}%`);
        console.log(`   All Predictions:`, detectionResult.allPredictions || []);
      } else {
        console.log(`   Error: ${detectionResult.error}`);
      }
      
      if (!detectionResult.success || !detectionResult.data) {
        throw new Error(detectionResult.error || 'Failed to detect disease');
      }

      console.log("\n🎯 USING YOUR MODEL RESULTS...");
      console.log(`   Disease Name: ${detectionResult.data.diseaseName}`);
      console.log(`   Confidence: ${detectionResult.data.confidence}%`);
      console.log("   Source: Your PyTorch Model");
      
      console.log("\n🤖 GETTING DETAILED DISEASE INFORMATION...");
      const diseaseInfo = await getDiseaseInfo(detectionResult.data.diseaseName);
      
      console.log("✅ Disease information retrieved successfully");
      console.log(`   Description: ${diseaseInfo.description.substring(0, 100)}...`);
      console.log(`   Symptoms: ${diseaseInfo.symptoms.length} items`);
      console.log(`   Treatment: ${diseaseInfo.treatment.length} items`);
      
      // Combine model results with detailed disease information
      const combinedResult: DiseaseResult = {
        ...detectionResult.data,
        description: diseaseInfo.description,
        symptoms: diseaseInfo.symptoms,
        causes: diseaseInfo.causes,
        treatment: diseaseInfo.treatment,
        prevention: diseaseInfo.prevention,
        severity: diseaseInfo.severity,
        affectedPlantParts: diseaseInfo.affectedPlantParts,
        imageUrl: URL.createObjectURL(image),
        allPredictions: detectionResult.allPredictions || []
      };

      console.log("\n🎯 FINAL COMBINED RESULT:");
      console.log(`   Disease: ${combinedResult.diseaseName}`);
      console.log(`   Confidence: ${combinedResult.confidence}%`);
      console.log(`   Severity: ${combinedResult.severity}`);
      console.log(`   Image URL: ${combinedResult.imageUrl}`);

      console.log("\n📤 SENDING RESULT TO PARENT COMPONENT...");
      onDiseaseDetected(combinedResult);
      
      console.log("✅ DISEASE DETECTION COMPLETED SUCCESSFULLY!");
      
    } catch (err) {
      console.log("\n❌ ERROR DURING DISEASE DETECTION:");
      console.log(`   Error: ${err instanceof Error ? err.message : 'Unknown error'}`);
      console.log(`   Stack: ${err instanceof Error ? err.stack : 'No stack trace'}`);
      
      setError(err instanceof Error ? err.message : 'An error occurred during analysis');
    } finally {
      setIsAnalyzing(false);
      console.log("=".repeat(80));
      console.log("🔍 FRONTEND: DISEASE DETECTION COMPLETE");
      console.log("=".repeat(80) + "\n");
    }
  };

  return (
    <div className="card">
      <div className="flex items-center space-x-3 mb-4">
        <div className="flex items-center justify-center w-10 h-10 bg-blue-100 rounded-lg">
          <Brain className="w-5 h-5 text-blue-600" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-gray-900">Disease Detection</h2>
          <p className="text-sm text-gray-500">AI-powered plant health analysis</p>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-5 h-5 text-red-500" />
            <p className="text-red-700">{error}</p>
          </div>
        </div>
      )}


      <div className="space-y-4">
        <div className="bg-gray-50 rounded-lg p-4">
          <h3 className="font-medium text-gray-900 mb-2">Ready to analyze</h3>
          <p className="text-sm text-gray-600">
            Click the button below to analyze your plant image for diseases using our AI model.
          </p>
        </div>

        <button
          onClick={handleAnalyze}
          disabled={isAnalyzing}
          className="w-full btn-primary flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isAnalyzing ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <Brain className="w-5 h-5" />
              <span>Analyze Plant Disease</span>
            </>
          )}
        </button>

        {isAnalyzing && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
              <span className="text-sm font-medium text-blue-800">Processing...</span>
            </div>
            <div className="text-sm text-blue-700">
              <p>1. Analyzing image with AI model...</p>
              <p>2. Identifying potential diseases...</p>
              <p>3. Gathering detailed information...</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DiseaseDetection;
