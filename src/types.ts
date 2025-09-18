export interface DiseaseResult {
  diseaseName: string;
  confidence: number;
  description: string;
  symptoms: string[];
  causes: string[];
  treatment: string[];
  prevention: string[];
  severity: 'Low' | 'Medium' | 'High' | 'Critical';
  affectedPlantParts: string[];
  imageUrl?: string;
  allPredictions?: ModelResponse[];
}

export interface ModelResponse {
  diseaseName: string;
  confidence: number;
}

export interface ApiResponse {
  success: boolean;
  data?: ModelResponse;
  error?: string;
  allPredictions?: ModelResponse[];
}

export interface GeminiApiResponse {
  diseaseName: string;
  description: string;
  symptoms: string[];
  causes: string[];
  treatment: string[];
  prevention: string[];
  severity: 'Low' | 'Medium' | 'High' | 'Critical';
  affectedPlantParts: string[];
}
