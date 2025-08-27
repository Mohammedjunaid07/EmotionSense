import cv2
import numpy as np
import streamlit as st
from pathlib import Path

class FaceEmotionAnalyzer:
    """Analyzes emotions from facial expressions using OpenCV and heuristic methods"""
    
    def __init__(self):
        self.emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear']
        self.face_cascade = None
        self.eye_cascade = None
        self.smile_cascade = None
        self._load_cascades()
    
    def _load_cascades(self):
        """Load OpenCV Haar cascades for face detection"""
        try:
            # Try to load pre-trained Haar cascades
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            
        except Exception as e:
            st.warning(f"Could not load OpenCV cascades: {str(e)}")
            self.face_cascade = None
            self.eye_cascade = None
            self.smile_cascade = None
    
    def _detect_facial_features(self, image):
        """Detect facial features in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        features = {
            'faces': [],
            'eyes': [],
            'smiles': [],
            'face_detected': False
        }
        
        if self.face_cascade is not None:
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            features['faces'] = faces
            features['face_detected'] = len(faces) > 0
            
            # For each face, detect eyes and smiles
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = image[y:y+h, x:x+w]
                
                if self.eye_cascade is not None:
                    eyes = self.eye_cascade.detectMultiScale(roi_gray)
                    features['eyes'].extend(eyes)
                
                if self.smile_cascade is not None:
                    smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                    features['smiles'].extend(smiles)
        
        return features
    
    def _analyze_facial_geometry(self, image):
        """Analyze facial geometry for emotion cues"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Basic geometric analysis
        features = {}
        
        # Calculate brightness in different regions (rough approximation)
        # Top third (forehead/eyebrows)
        top_region = gray[:h//3, :]
        features['top_brightness'] = np.mean(top_region)
        
        # Middle third (eyes/nose)
        middle_region = gray[h//3:2*h//3, :]
        features['middle_brightness'] = np.mean(middle_region)
        
        # Bottom third (mouth area)
        bottom_region = gray[2*h//3:, :]
        features['bottom_brightness'] = np.mean(bottom_region)
        
        # Edge density (can indicate facial expressions)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (h * w)
        
        # Contrast measures
        features['contrast'] = np.std(gray)
        
        return features
    
    def _predict_emotion_from_features(self, facial_features, geometric_features):
        """Predict emotion based on detected features"""
        scores = {emotion: 0.0 for emotion in self.emotions}
        
        # Default to neutral if no face detected
        if not facial_features['face_detected']:
            scores['neutral'] = 0.7
            scores['sad'] = 0.1
            scores['fear'] = 0.1
            scores['angry'] = 0.1
            return scores
        
        # Number of detected features
        num_faces = len(facial_features['faces'])
        num_eyes = len(facial_features['eyes'])
        num_smiles = len(facial_features['smiles'])
        
        # Happy emotion indicators
        if num_smiles > 0:
            scores['happy'] += 0.5
            scores['surprise'] += 0.1
        
        # Surprise indicators (wide eyes, etc.)
        if num_eyes > 2:  # More eyes detected might indicate wide open eyes
            scores['surprise'] += 0.2
            scores['fear'] += 0.1
        
        # Use geometric features for additional emotion cues
        if geometric_features:
            # High contrast might indicate strong emotions
            if geometric_features['contrast'] > 50:
                scores['angry'] += 0.2
                scores['surprise'] += 0.1
            
            # Low contrast might indicate sadness
            if geometric_features['contrast'] < 30:
                scores['sad'] += 0.2
                scores['neutral'] += 0.1
            
            # Bottom region brightness (mouth area)
            bottom_brightness = geometric_features['bottom_brightness']
            if bottom_brightness > geometric_features['middle_brightness'] * 1.1:
                scores['happy'] += 0.1  # Bright mouth area might indicate smile
            elif bottom_brightness < geometric_features['middle_brightness'] * 0.9:
                scores['sad'] += 0.1  # Dark mouth area might indicate frown
            
            # Edge density
            edge_density = geometric_features['edge_density']
            if edge_density > 0.1:
                scores['angry'] += 0.1
                scores['surprise'] += 0.1
            
        # Add some baseline neutral emotion
        scores['neutral'] += 0.2
        
        # Add randomness based on features to avoid deterministic results
        feature_sum = sum([num_faces, num_eyes, num_smiles])
        if feature_sum > 0:
            np.random.seed(feature_sum * 17)  # Deterministic but feature-dependent
            noise = np.random.random(len(self.emotions)) * 0.1
            for i, emotion in enumerate(self.emotions):
                scores[emotion] += noise[i]
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            scores['neutral'] = 1.0
        
        return scores
    
    def analyze(self, image_path):
        """
        Analyze emotions in the given image
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            dict: Dictionary with emotion scores
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            
            if image is None:
                st.error("Could not load image file")
                return {emotion: 1.0/len(self.emotions) for emotion in self.emotions}
            
            # Detect facial features
            facial_features = self._detect_facial_features(image)
            
            # Analyze facial geometry
            geometric_features = self._analyze_facial_geometry(image)
            
            # Predict emotions
            emotions = self._predict_emotion_from_features(facial_features, geometric_features)
            
            return emotions
            
        except Exception as e:
            st.error(f"Error analyzing facial expression: {str(e)}")
            return {emotion: 1.0/len(self.emotions) for emotion in self.emotions}
    
    def get_dominant_emotion(self, emotions_dict):
        """Get the emotion with highest confidence"""
        return max(emotions_dict.items(), key=lambda x: x[1])
