import React, { useCallback, useState } from 'react';
import { Upload, X, Image as ImageIcon } from 'lucide-react';

interface ImageUploadProps {
  onImageUpload: (file: File) => void;
  uploadedImage: File | null;
  onReset: () => void;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ onImageUpload, uploadedImage, onReset }) => {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    const imageFile = files.find(file => file.type.startsWith('image/'));
    
    if (imageFile) {
      onImageUpload(imageFile);
    }
  }, [onImageUpload]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      onImageUpload(file);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  if (uploadedImage) {
    return (
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Uploaded Image</h2>
          <button
            onClick={onReset}
            className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        
        <div className="space-y-4">
          <div className="relative">
            <img
              src={URL.createObjectURL(uploadedImage)}
              alt="Uploaded plant"
              className="w-full h-64 object-cover rounded-lg border border-gray-200"
            />
          </div>
          
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <ImageIcon className="w-4 h-4 text-gray-500" />
              <span className="text-sm font-medium text-gray-700">{uploadedImage.name}</span>
            </div>
            <div className="text-sm text-gray-500">
              Size: {formatFileSize(uploadedImage.size)}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">Upload Plant Image</h2>
      
      <div
        className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          isDragOver
            ? 'border-primary-400 bg-primary-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        
        <div className="space-y-4">
          <div className="flex justify-center">
            <div className="flex items-center justify-center w-16 h-16 bg-primary-100 rounded-full">
              <Upload className="w-8 h-8 text-primary-600" />
            </div>
          </div>
          
          <div>
            <p className="text-lg font-medium text-gray-900 mb-2">
              Drop your plant image here
            </p>
            <p className="text-gray-500 mb-4">
              or click to browse files
            </p>
            <p className="text-sm text-gray-400">
              Supports JPG, PNG, GIF up to 10MB
            </p>
          </div>
        </div>
      </div>
      
      <div className="mt-4 text-sm text-gray-500">
        <p className="font-medium mb-1">Tips for best results:</p>
        <ul className="list-disc list-inside space-y-1">
          <li>Use clear, well-lit photos</li>
          <li>Focus on affected plant parts</li>
          <li>Include leaves, stems, or flowers in the image</li>
          <li>Avoid blurry or dark images</li>
        </ul>
      </div>
    </div>
  );
};

export default ImageUpload;
