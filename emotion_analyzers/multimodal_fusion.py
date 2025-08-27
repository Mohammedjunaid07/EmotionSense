import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st

class MultimodalFusion:
    """Fuses emotion predictions from multiple modalities"""
    
    def __init__(self):
        self.emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear']
        self.modality_weights = {
            'text': 0.4,
            'audio': 0.35,
            'face': 0.25
        }
    
    def _normalize_predictions(self, predictions):
        """Normalize prediction scores to sum to 1"""
        if not predictions:
            return {}
        
        total = sum(predictions.values())
        if total > 0:
            return {k: v/total for k, v in predictions.items()}
        else:
            return {emotion: 1.0/len(self.emotions) for emotion in self.emotions}
    
    def _weighted_average_fusion(self, text_emotions, audio_emotions, face_emotions):
        """Combine predictions using weighted average"""
        fused_emotions = {emotion: 0.0 for emotion in self.emotions}
        total_weight = 0.0
        
        # Add text emotions
        if text_emotions:
            normalized_text = self._normalize_predictions(text_emotions)
            weight = self.modality_weights['text']
            for emotion in self.emotions:
                fused_emotions[emotion] += normalized_text.get(emotion, 0) * weight
            total_weight += weight
        
        # Add audio emotions
        if audio_emotions:
            normalized_audio = self._normalize_predictions(audio_emotions)
            weight = self.modality_weights['audio']
            for emotion in self.emotions:
                fused_emotions[emotion] += normalized_audio.get(emotion, 0) * weight
            total_weight += weight
        
        # Add face emotions
        if face_emotions:
            normalized_face = self._normalize_predictions(face_emotions)
            weight = self.modality_weights['face']
            for emotion in self.emotions:
                fused_emotions[emotion] += normalized_face.get(emotion, 0) * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            fused_emotions = {k: v/total_weight for k, v in fused_emotions.items()}
        
        return fused_emotions
    
    def _majority_voting_fusion(self, text_emotions, audio_emotions, face_emotions):
        """Combine predictions using majority voting on dominant emotions"""
        predictions = []
        
        if text_emotions:
            dominant_text = max(text_emotions.items(), key=lambda x: x[1])
            predictions.append(dominant_text[0])
        
        if audio_emotions:
            dominant_audio = max(audio_emotions.items(), key=lambda x: x[1])
            predictions.append(dominant_audio[0])
        
        if face_emotions:
            dominant_face = max(face_emotions.items(), key=lambda x: x[1])
            predictions.append(dominant_face[0])
        
        if not predictions:
            return {emotion: 1.0/len(self.emotions) for emotion in self.emotions}
        
        # Count votes for each emotion
        emotion_votes = {emotion: 0 for emotion in self.emotions}
        for pred in predictions:
            emotion_votes[pred] += 1
        
        # Convert votes to probabilities
        total_votes = len(predictions)
        emotion_probs = {k: v/total_votes for k, v in emotion_votes.items()}
        
        return emotion_probs
    
    def _confidence_weighted_fusion(self, text_emotions, audio_emotions, face_emotions):
        """Combine predictions using confidence-weighted fusion"""
        fused_emotions = {emotion: 0.0 for emotion in self.emotions}
        
        # Calculate confidence as max probability for each modality
        confidences = {}
        predictions = {}
        
        if text_emotions:
            text_normalized = self._normalize_predictions(text_emotions)
            confidences['text'] = max(text_normalized.values())
            predictions['text'] = text_normalized
        
        if audio_emotions:
            audio_normalized = self._normalize_predictions(audio_emotions)
            confidences['audio'] = max(audio_normalized.values())
            predictions['audio'] = audio_normalized
        
        if face_emotions:
            face_normalized = self._normalize_predictions(face_emotions)
            confidences['face'] = max(face_normalized.values())
            predictions['face'] = face_normalized
        
        # Weight by confidence
        total_confidence = sum(confidences.values())
        
        if total_confidence > 0:
            for modality, pred in predictions.items():
                weight = confidences[modality] / total_confidence
                for emotion in self.emotions:
                    fused_emotions[emotion] += pred.get(emotion, 0) * weight
        
        return fused_emotions
    
    def fuse_predictions(self, text_emotions=None, audio_emotions=None, face_emotions=None, 
                        method='weighted_average'):
        """
        Fuse emotion predictions from multiple modalities
        
        Args:
            text_emotions (dict): Text emotion predictions
            audio_emotions (dict): Audio emotion predictions 
            face_emotions (dict): Face emotion predictions
            method (str): Fusion method ('weighted_average', 'majority_voting', 'confidence_weighted')
            
        Returns:
            dict: Fused emotion predictions
        """
        
        # Check if we have at least one prediction
        if not any([text_emotions, audio_emotions, face_emotions]):
            return {emotion: 1.0/len(self.emotions) for emotion in self.emotions}
        
        try:
            if method == 'weighted_average':
                fused = self._weighted_average_fusion(text_emotions, audio_emotions, face_emotions)
            elif method == 'majority_voting':
                fused = self._majority_voting_fusion(text_emotions, audio_emotions, face_emotions)
            elif method == 'confidence_weighted':
                fused = self._confidence_weighted_fusion(text_emotions, audio_emotions, face_emotions)
            else:
                # Default to weighted average
                fused = self._weighted_average_fusion(text_emotions, audio_emotions, face_emotions)
            
            # Ensure all emotions are present and normalized
            total = sum(fused.values())
            if total > 0:
                fused = {k: v/total for k, v in fused.items()}
            
            return fused
            
        except Exception as e:
            st.error(f"Error in multimodal fusion: {str(e)}")
            return {emotion: 1.0/len(self.emotions) for emotion in self.emotions}
    
    def get_fusion_confidence(self, fused_emotions):
        """Calculate overall confidence of fused prediction"""
        if not fused_emotions:
            return 0.0
        
        # Confidence as max probability
        max_prob = max(fused_emotions.values())
        
        # Entropy-based confidence (lower entropy = higher confidence)
        entropy = -sum(p * np.log2(p + 1e-10) for p in fused_emotions.values() if p > 0)
        max_entropy = np.log2(len(self.emotions))
        normalized_entropy = entropy / max_entropy
        entropy_confidence = 1 - normalized_entropy
        
        # Combined confidence
        confidence = (max_prob + entropy_confidence) / 2
        
        return confidence
    
    def get_dominant_emotion(self, emotions_dict):
        """Get the emotion with highest confidence"""
        if not emotions_dict:
            return ('neutral', 0.0)
        return max(emotions_dict.items(), key=lambda x: x[1])
