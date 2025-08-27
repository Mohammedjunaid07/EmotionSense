import numpy as np
import streamlit as st
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    # Create dummy classes to avoid undefined variable errors
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name): return None
    class AutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(name): return None
    def pipeline(*args, **kwargs): return None

class TextEmotionAnalyzer:
    """Analyzes emotions from text using pre-trained transformer models"""
    
    def __init__(self):
        self.emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear']
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained emotion classification model"""
        if not HAS_TRANSFORMERS:
            # Don't show the info message every time, just use fallback
            self.classifier = None
            return
            
        try:
            # Using a lightweight emotion classification model
            model_name = "j-hartmann/emotion-english-distilroberta-base"
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Create pipeline
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1  # Use CPU
            )
            
        except Exception as e:
            st.warning(f"Could not load transformer model, using rule-based analysis: {str(e)}")
            # Fallback to a simpler model
            try:
                self.classifier = pipeline(
                    "text-classification",
                    model="cardiffnlp/twitter-roberta-base-emotion",
                    device=-1
                )
            except:
                # Final fallback - create a simple rule-based classifier
                self.classifier = None
    
    def _map_emotions(self, predictions):
        """Map model predictions to our standard emotion set"""
        emotion_mapping = {
            'joy': 'happy',
            'happiness': 'happy',
            'sadness': 'sad',
            'anger': 'angry',
            'fear': 'fear',
            'surprise': 'surprise',
            'neutral': 'neutral',
            'disgust': 'angry',  # Map disgust to angry
            'love': 'happy',     # Map love to happy
            'optimism': 'happy', # Map optimism to happy
            'pessimism': 'sad'   # Map pessimism to sad
        }
        
        # Initialize result with zeros
        result = {emotion: 0.0 for emotion in self.emotions}
        
        # Process predictions
        for pred in predictions:
            label = pred['label'].lower()
            score = pred['score']
            
            # Map to our emotion set
            mapped_emotion = emotion_mapping.get(label, 'neutral')
            result[mapped_emotion] += score
        
        # Normalize to ensure sum equals 1
        total = sum(result.values())
        if total > 0:
            result = {k: v/total for k, v in result.items()}
        else:
            # If no emotions detected, default to neutral
            result['neutral'] = 1.0
        
        return result
    
    def _simple_rule_based_analysis(self, text):
        """Simple rule-based emotion analysis as fallback"""
        text_lower = text.lower()
        
        # Expanded keyword-based emotion detection
        emotion_keywords = {
            'happy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic', 'love', 'smile', 'laugh', 
                     'cheerful', 'delighted', 'pleased', 'thrilled', 'good', 'excellent', 'awesome', 'brilliant',
                     'beautiful', 'perfect', 'celebration', 'celebrate', 'fun', 'enjoy', 'glad', 'blessed'],
            'sad': ['sad', 'depressed', 'unhappy', 'disappointed', 'cry', 'tears', 'sorrow', 'grief', 
                   'down', 'blue', 'upset', 'hurt', 'pain', 'suffer', 'broken', 'lonely', 'empty',
                   'devastated', 'heartbroken', 'miserable', 'gloom', 'despair', 'melancholy'],
            'angry': ['angry', 'mad', 'furious', 'hate', 'rage', 'annoyed', 'frustrated', 'irritated',
                     'outraged', 'livid', 'enraged', 'pissed', 'fed up', 'disgusted', 'resentful',
                     'bitter', 'hostile', 'aggressive', 'violent', 'furious', 'irate'],
            'fear': ['afraid', 'scared', 'frightened', 'terrified', 'worry', 'anxious', 'nervous', 'panic',
                    'fearful', 'paranoid', 'stressed', 'concerned', 'uneasy', 'apprehensive', 'dread',
                    'horror', 'terror', 'phobia', 'alarmed', 'threatened', 'insecure'],
            'surprise': ['surprised', 'amazed', 'shocked', 'astonished', 'wow', 'incredible', 'unbelievable',
                        'stunning', 'remarkable', 'extraordinary', 'unexpected', 'sudden', 'startled',
                        'bewildered', 'speechless', 'mind-blown', 'overwhelmed', 'awestruck'],
            'neutral': ['okay', 'fine', 'normal', 'regular', 'standard', 'average', 'typical', 'usual',
                       'moderate', 'calm', 'peaceful', 'stable', 'balanced', 'routine', 'ordinary']
        }
        
        scores = {emotion: 0.0 for emotion in self.emotions}
        total_words = len(text_lower.split())
        
        # Count keyword matches with weight
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Weight longer keywords more heavily
                    weight = len(keyword) / 10.0 + 1.0
                    scores[emotion] += weight
        
        # Add sentiment intensity based on punctuation and caps
        if '!' in text:
            # Exclamation marks suggest stronger emotion
            max_emotion = max(scores.items(), key=lambda x: x[1])
            if max_emotion[1] > 0:
                scores[max_emotion[0]] *= 1.3
        
        if text.isupper() and len(text) > 3:
            # ALL CAPS suggests strong emotion
            max_emotion = max(scores.items(), key=lambda x: x[1])
            if max_emotion[1] > 0:
                scores[max_emotion[0]] *= 1.5
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            # If no emotion words found, assign some neutral probability
            scores['neutral'] = 0.7
            scores['happy'] = 0.1
            scores['sad'] = 0.1
            scores['angry'] = 0.05
            scores['fear'] = 0.03
            scores['surprise'] = 0.02
        
        return scores
    
    def analyze(self, text):
        """
        Analyze emotions in the given text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary with emotion scores
        """
        if not text or not text.strip():
            return {emotion: 0.0 for emotion in self.emotions}
        
        try:
            if self.classifier is not None:
                # Use transformer model
                predictions = self.classifier(text)
                
                # Handle both single prediction and list of predictions
                if isinstance(predictions, dict):
                    predictions = [predictions]
                
                return self._map_emotions(predictions)
            else:
                # Use rule-based fallback
                return self._simple_rule_based_analysis(text)
                
        except Exception as e:
            st.warning(f"Error in text analysis, using fallback method: {str(e)}")
            return self._simple_rule_based_analysis(text)
    
    def get_dominant_emotion(self, emotions_dict):
        """Get the emotion with highest confidence"""
        return max(emotions_dict.items(), key=lambda x: x[1])
