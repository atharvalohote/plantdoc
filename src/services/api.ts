import { ApiResponse, GeminiApiResponse } from '../types';

// Configuration - Update these with your actual API endpoints
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
const GEMINI_API_KEY = process.env.REACT_APP_GEMINI_API_KEY || ''; // Add your Gemini API key here

/**
 * Detect disease using your custom model
 */
export const detectDisease = async (image: File): Promise<ApiResponse> => {
  console.log("\n" + "=".repeat(60));
  console.log("🌐 API SERVICE: DISEASE DETECTION REQUEST");
  console.log("=".repeat(60));
  
  try {
    console.log("📤 PREPARING REQUEST:");
    console.log(`   URL: ${API_BASE_URL}/api/detect-disease`);
    console.log(`   Method: POST`);
    console.log(`   Image: ${image.name} (${image.size} bytes)`);
    
    const formData = new FormData();
    formData.append('image', image);
    
    console.log("📡 SENDING REQUEST TO BACKEND...");
    const startTime = Date.now();
    
    const response = await fetch(`${API_BASE_URL}/api/detect-disease`, {
      method: 'POST',
      body: formData,
    });
    
    const endTime = Date.now();
    const duration = endTime - startTime;
    
    console.log("📥 RESPONSE RECEIVED:");
    console.log(`   Status: ${response.status} ${response.statusText}`);
    console.log(`   Duration: ${duration}ms`);
    console.log(`   Content-Type: ${response.headers.get('content-type')}`);

    if (!response.ok) {
      const errorText = await response.text();
      console.log(`   Error Response: ${errorText}`);
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    
    console.log("📊 RESPONSE DATA:");
    console.log(`   Success: ${result.success}`);
    if (result.success && result.data) {
      console.log(`   Disease: ${result.data.diseaseName}`);
      console.log(`   Confidence: ${result.data.confidence}%`);
    } else {
      console.log(`   Error: ${result.error}`);
    }
    
    console.log("✅ API REQUEST COMPLETED SUCCESSFULLY");
    return result;
    
  } catch (error) {
    console.log("\n❌ API REQUEST FAILED:");
    console.log(`   Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    console.log(`   Type: ${error instanceof Error ? error.constructor.name : typeof error}`);
    
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Failed to detect disease'
    };
  } finally {
    console.log("=".repeat(60));
    console.log("🌐 API SERVICE: REQUEST COMPLETE");
    console.log("=".repeat(60) + "\n");
  }
};

/**
 * Get detailed disease information using Gemini AI
 */
export const getDiseaseInfo = async (diseaseName: string): Promise<GeminiApiResponse> => {
  console.log("\n" + "=".repeat(60));
  console.log("🤖 GEMINI AI: GETTING DISEASE INFORMATION");
  console.log("=".repeat(60));
  console.log(`📋 Disease: ${diseaseName}`);
  
  try {
    if (!GEMINI_API_KEY) {
      console.log("⚠️  No Gemini API key found, using fallback information");
      return getFallbackDiseaseInfo(diseaseName);
    }

    console.log("🚀 Calling Gemini API for detailed disease information...");
    
    const prompt = `You are a plant disease expert. Provide detailed information about "${diseaseName}" in the following JSON format:

{
  "diseaseName": "${diseaseName}",
  "description": "Detailed description of the disease",
  "symptoms": ["symptom1", "symptom2", "symptom3"],
  "causes": ["cause1", "cause2", "cause3"],
  "treatment": ["treatment1", "treatment2", "treatment3"],
  "prevention": ["prevention1", "prevention2", "prevention3"],
  "severity": "Low|Medium|High|Critical",
  "affectedPlantParts": ["part1", "part2", "part3"]
}

Make sure the information is accurate, scientific, and helpful for plant care.`;

    const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_API_KEY}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [{
          parts: [{
            text: prompt
          }]
        }]
      })
    });

    if (!response.ok) {
      console.log(`❌ Gemini API error: ${response.status}`);
      return getFallbackDiseaseInfo(diseaseName);
    }

    const data = await response.json();
    console.log("✅ Gemini API response received");
    
    if (data.candidates && data.candidates[0] && data.candidates[0].content) {
      const content = data.candidates[0].content.parts[0].text;
      console.log("📝 Parsing Gemini response...");
      
      try {
        // Extract JSON from the response
        const jsonMatch = content.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          const diseaseInfo = JSON.parse(jsonMatch[0]);
          console.log("✅ Successfully parsed disease information from Gemini");
          return diseaseInfo;
        }
      } catch (parseError) {
        console.log("⚠️  Failed to parse Gemini response, using fallback");
      }
    }
    
    return getFallbackDiseaseInfo(diseaseName);
    
  } catch (error) {
    console.log(`❌ Gemini API error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    return getFallbackDiseaseInfo(diseaseName);
  } finally {
    console.log("=".repeat(60));
    console.log("🤖 GEMINI AI: REQUEST COMPLETE");
    console.log("=".repeat(60) + "\n");
  }
};

/**
 * Fallback disease information when Gemini is not available
 */
const getFallbackDiseaseInfo = (diseaseName: string): GeminiApiResponse => {
  console.log("🔄 Using fallback disease information");
  
  const isHealthy = diseaseName.toLowerCase().includes('healthy');
  
  return {
    diseaseName: diseaseName,
    description: isHealthy 
      ? `Your AI model detected this plant as healthy with no disease symptoms.`
      : `Your AI model detected: ${diseaseName}. This is a real prediction from your trained model.`,
    symptoms: isHealthy 
      ? ["No disease symptoms detected", "Healthy plant appearance", "Normal growth patterns"]
      : ["Disease symptoms detected by AI model", "Visual signs may be present", "Monitor plant health closely"],
    causes: isHealthy
      ? ["Good plant care practices", "Healthy growing conditions", "No disease pathogens present"]
      : ["Environmental stress factors", "Pathogen presence", "Plant health conditions"],
    treatment: isHealthy
      ? ["Continue current care routine", "Maintain proper watering", "Ensure adequate nutrition"]
      : ["Consult plant disease resources", "Apply appropriate treatment", "Monitor plant recovery"],
    prevention: isHealthy
      ? ["Continue good care practices", "Regular plant monitoring", "Maintain healthy environment"]
      : ["Improve plant care practices", "Regular health monitoring", "Preventive measures"],
    severity: isHealthy ? 'Low' : 'Medium',
    affectedPlantParts: isHealthy ? ["None - plant is healthy"] : ["Leaves", "Stems", "Roots"]
  };
};

/**
 * Health check for the API
 */
export const healthCheck = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch (error) {
    console.error('API health check failed:', error);
    return false;
  }
};