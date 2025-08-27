# Overview

This is a multimodal emotion analysis application built with Streamlit that analyzes emotions from text, audio, and facial expressions. The system uses machine learning models to detect emotions across different input modalities and provides a fusion mechanism to combine predictions for more accurate emotion recognition. The application features an interactive web interface with support for file uploads, live webcam capture, and real-time emotion analysis. Results are visualized through interactive charts and radar plots.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
The application uses Streamlit as the web framework, providing an interactive interface for file uploads and real-time emotion analysis. The UI includes visualization components using Plotly for creating bar charts and radar charts to display emotion predictions across different modalities.

## Backend Architecture
The system follows a modular architecture with separate analyzer classes for each modality:

- **Text Analysis**: Uses pre-trained transformer models (DistilRoBERTa) from Hugging Face for emotion classification from text input
- **Audio Analysis**: Extracts acoustic features using librosa (MFCC, chroma, mel spectrogram, spectral features) and applies machine learning models for emotion detection
- **Face Analysis**: Utilizes OpenCV Haar cascades for facial feature detection and heuristic methods for emotion recognition
- **Multimodal Fusion**: Implements weighted average fusion to combine predictions from multiple modalities with configurable weights

## Data Processing Pipeline
The system processes different file types (text, audio, images/video) through specialized analyzers. Each analyzer extracts relevant features and generates emotion predictions that are then normalized and combined through the fusion module.

## Caching Strategy
Uses Streamlit's `@st.cache_resource` decorator to cache loaded models and analyzers, improving performance by avoiding repeated model loading.

## Error Handling
Implements fallback mechanisms for model loading failures and provides graceful degradation when certain components (like OpenCV cascades) fail to load.

# External Dependencies

## Machine Learning Libraries
- **Transformers**: Hugging Face library for pre-trained emotion classification models
- **PyTorch**: Backend for transformer models
- **Scikit-learn**: For preprocessing and scaling features
- **Joblib**: For model serialization

## Audio Processing
- **librosa**: Audio analysis and feature extraction library for MFCC, chroma, and spectral features

## Computer Vision
- **OpenCV**: Face detection using Haar cascades and image processing

## Data Science Stack
- **NumPy**: Numerical computing for array operations
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualization for emotion charts and radar plots

## Web Framework
- **Streamlit**: Web application framework for the user interface

## File Processing
- **tempfile**: Temporary file handling for uploaded files
- **pathlib**: File path operations and validation