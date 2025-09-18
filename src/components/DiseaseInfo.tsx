import React, { useState } from 'react';
import { 
  Info, 
  AlertTriangle, 
  CheckCircle, 
  Shield, 
  ChevronDown, 
  ChevronUp,
  ExternalLink,
  BarChart3,
  Target
} from 'lucide-react';
import { DiseaseResult } from '../types';

interface DiseaseInfoProps {
  diseaseResult: DiseaseResult;
}

const DiseaseInfo: React.FC<DiseaseInfoProps> = ({ diseaseResult }) => {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['overview']));

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'Low': return 'text-green-600 bg-green-100';
      case 'Medium': return 'text-yellow-600 bg-yellow-100';
      case 'High': return 'text-orange-600 bg-orange-100';
      case 'Critical': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'text-green-600';
    if (confidence >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const SectionHeader: React.FC<{ 
    title: string; 
    icon: React.ReactNode; 
    section: string; 
    isExpanded: boolean;
  }> = ({ title, icon, section, isExpanded }) => (
    <button
      onClick={() => toggleSection(section)}
      className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50 transition-colors"
    >
      <div className="flex items-center space-x-3">
        {icon}
        <h3 className="font-semibold text-gray-900">{title}</h3>
      </div>
      {isExpanded ? (
        <ChevronUp className="w-5 h-5 text-gray-400" />
      ) : (
        <ChevronDown className="w-5 h-5 text-gray-400" />
      )}
    </button>
  );

  return (
    <div className="card">
      <div className="flex items-center space-x-3 mb-6">
        <div className="flex items-center justify-center w-10 h-10 bg-green-100 rounded-lg">
          <Info className="w-5 h-5 text-green-600" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-gray-900">Disease Information</h2>
          <p className="text-sm text-gray-500">Detailed analysis and recommendations</p>
        </div>
      </div>

      {/* Disease Overview */}
      <div className="mb-6">
        <div className="bg-gradient-to-r from-primary-50 to-secondary-50 rounded-lg p-6">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">
                {diseaseResult.diseaseName}
              </h3>
              <p className="text-gray-600 mb-4">{diseaseResult.description}</p>
            </div>
            <div className="flex flex-col items-end space-y-2">
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${getSeverityColor(diseaseResult.severity)}`}>
                {diseaseResult.severity} Severity
              </span>
              <span className={`text-sm font-medium ${getConfidenceColor(diseaseResult.confidence)}`}>
                {diseaseResult.confidence}% Confidence
              </span>
            </div>
          </div>
          
          {diseaseResult.affectedPlantParts.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {diseaseResult.affectedPlantParts.map((part, index) => (
                <span
                  key={index}
                  className="px-2 py-1 bg-white rounded-md text-sm text-gray-600 border border-gray-200"
                >
                  {part}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Model Analysis Section */}
      <div className="mb-6">
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4">
          <div className="flex items-center space-x-3 mb-3">
            <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
              <BarChart3 className="w-4 h-4 text-blue-600" />
            </div>
            <h3 className="font-semibold text-gray-900">Model Analysis</h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className={`text-2xl font-bold ${
                diseaseResult.confidence >= 80 ? 'text-green-600' : 
                diseaseResult.confidence >= 60 ? 'text-yellow-600' : 'text-red-600'
              }`}>
                {diseaseResult.confidence}%
              </div>
              <div className="text-sm text-gray-600">Confidence Score</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {diseaseResult.allPredictions?.length || 0}
              </div>
              <div className="text-sm text-gray-600">Total Predictions</div>
            </div>
            
            <div className="text-center">
              <div className={`text-2xl font-bold ${
                diseaseResult.severity === 'High' ? 'text-red-600' :
                diseaseResult.severity === 'Medium' ? 'text-yellow-600' : 'text-green-600'
              }`}>
                {diseaseResult.severity}
              </div>
              <div className="text-sm text-gray-600">Severity Level</div>
            </div>
          </div>
          
          <div className="mt-3 text-xs text-gray-600">
            <p><strong>Analysis:</strong> Your PyTorch model processed the image and provided {diseaseResult.allPredictions?.length || 0} disease predictions. 
            {diseaseResult.confidence >= 80 ? ' High confidence indicates strong model certainty.' : 
             diseaseResult.confidence >= 60 ? ' Medium confidence suggests good model certainty.' : 
             ' Lower confidence indicates the model is less certain about this prediction.'}
            </p>
          </div>
        </div>
      </div>

      {/* All Predictions Section */}
      {diseaseResult.allPredictions && diseaseResult.allPredictions.length > 0 && (
        <div className="mb-6">
          <div className="border border-gray-200 rounded-lg">
            <SectionHeader
              title="All Model Predictions"
              icon={<BarChart3 className="w-5 h-5 text-blue-500" />}
              section="predictions"
              isExpanded={expandedSections.has('predictions')}
            />
            {expandedSections.has('predictions') && (
              <div className="px-4 pb-4">
                <div className="space-y-3">
                  {diseaseResult.allPredictions.map((prediction, index) => (
                    <div 
                      key={index} 
                      className={`p-3 rounded-lg border ${
                        index === 0 
                          ? 'bg-green-50 border-green-200' 
                          : 'bg-gray-50 border-gray-200'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                            index === 0 
                              ? 'bg-green-100 text-green-600' 
                              : 'bg-gray-100 text-gray-600'
                          }`}>
                            <Target className="w-4 h-4" />
                          </div>
                          <div>
                            <h4 className={`font-medium ${
                              index === 0 ? 'text-green-800' : 'text-gray-800'
                            }`}>
                              {prediction.diseaseName}
                            </h4>
                            <p className={`text-sm ${
                              index === 0 ? 'text-green-600' : 'text-gray-600'
                            }`}>
                              {index === 0 ? 'Top Prediction' : `Alternative ${index}`}
                            </p>
                          </div>
                        </div>
                        <div className={`text-right ${
                          index === 0 ? 'text-green-700' : 'text-gray-700'
                        }`}>
                          <div className={`text-lg font-bold ${
                            index === 0 ? 'text-green-600' : 'text-gray-600'
                          }`}>
                            {prediction.confidence}%
                          </div>
                          <div className="text-xs">Confidence</div>
                        </div>
                      </div>
                      {index === 0 && (
                        <div className="mt-2 text-xs text-green-600 font-medium">
                          ✅ This is the model's top prediction
                        </div>
                      )}
                    </div>
                  ))}
                </div>
                <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                  <div className="flex items-start space-x-2">
                    <Info className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                    <div className="text-sm text-blue-800">
                      <p className="font-medium mb-1">About These Predictions:</p>
                      <p>Your PyTorch model analyzed the image and ranked all possible diseases. The top prediction is the most likely, but other predictions may also be relevant depending on the plant type and symptoms.</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Expandable Sections */}
      <div className="space-y-2">
        {/* Symptoms */}
        <div className="border border-gray-200 rounded-lg">
          <SectionHeader
            title="Symptoms"
            icon={<AlertTriangle className="w-5 h-5 text-orange-500" />}
            section="symptoms"
            isExpanded={expandedSections.has('symptoms')}
          />
          {expandedSections.has('symptoms') && (
            <div className="px-4 pb-4">
              <ul className="space-y-2">
                {diseaseResult.symptoms.map((symptom, index) => (
                  <li key={index} className="flex items-start space-x-2">
                    <div className="w-1.5 h-1.5 bg-orange-500 rounded-full mt-2 flex-shrink-0" />
                    <span className="text-gray-700">{symptom}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Causes */}
        <div className="border border-gray-200 rounded-lg">
          <SectionHeader
            title="Causes"
            icon={<Info className="w-5 h-5 text-blue-500" />}
            section="causes"
            isExpanded={expandedSections.has('causes')}
          />
          {expandedSections.has('causes') && (
            <div className="px-4 pb-4">
              <ul className="space-y-2">
                {diseaseResult.causes.map((cause, index) => (
                  <li key={index} className="flex items-start space-x-2">
                    <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2 flex-shrink-0" />
                    <span className="text-gray-700">{cause}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Treatment */}
        <div className="border border-gray-200 rounded-lg">
          <SectionHeader
            title="Treatment"
            icon={<CheckCircle className="w-5 h-5 text-green-500" />}
            section="treatment"
            isExpanded={expandedSections.has('treatment')}
          />
          {expandedSections.has('treatment') && (
            <div className="px-4 pb-4">
              <ul className="space-y-2">
                {diseaseResult.treatment.map((treatment, index) => (
                  <li key={index} className="flex items-start space-x-2">
                    <div className="w-1.5 h-1.5 bg-green-500 rounded-full mt-2 flex-shrink-0" />
                    <span className="text-gray-700">{treatment}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Prevention */}
        <div className="border border-gray-200 rounded-lg">
          <SectionHeader
            title="Prevention"
            icon={<Shield className="w-5 h-5 text-purple-500" />}
            section="prevention"
            isExpanded={expandedSections.has('prevention')}
          />
          {expandedSections.has('prevention') && (
            <div className="px-4 pb-4">
              <ul className="space-y-2">
                {diseaseResult.prevention.map((prevention, index) => (
                  <li key={index} className="flex items-start space-x-2">
                    <div className="w-1.5 h-1.5 bg-purple-500 rounded-full mt-2 flex-shrink-0" />
                    <span className="text-gray-700">{prevention}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* Additional Resources */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <div className="flex items-center space-x-2 mb-2">
          <ExternalLink className="w-4 h-4 text-gray-500" />
          <span className="text-sm font-medium text-gray-700">Additional Resources</span>
        </div>
        <p className="text-sm text-gray-600">
          For more detailed information about {diseaseResult.diseaseName}, consult with a local plant pathologist or agricultural extension service.
        </p>
      </div>
    </div>
  );
};

export default DiseaseInfo;
