import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
from io import BytesIO

from emotion_analyzers.text_analyzer import TextEmotionAnalyzer
from emotion_analyzers.audio_analyzer import AudioEmotionAnalyzer
from emotion_analyzers.face_analyzer import FaceEmotionAnalyzer
from emotion_analyzers.multimodal_fusion import MultimodalFusion
from utils.file_utils import validate_file_type, save_uploaded_file, get_file_size_mb

# Initialize analyzers
@st.cache_resource
def load_analyzers():
    """Load and cache all emotion analyzers"""
    text_analyzer = TextEmotionAnalyzer()
    audio_analyzer = AudioEmotionAnalyzer()
    face_analyzer = FaceEmotionAnalyzer()
    fusion = MultimodalFusion()
    return text_analyzer, audio_analyzer, face_analyzer, fusion

def create_emotion_chart(emotions_dict, title):
    """Create a bar chart for emotion predictions"""
    emotions = list(emotions_dict.keys())
    scores = list(emotions_dict.values())
    
    fig = px.bar(
        x=emotions, 
        y=scores,
        title=title,
        labels={'x': 'Emotions', 'y': 'Confidence Score'},
        color=scores,
        color_continuous_scale='viridis'
    )
    fig.update_layout(showlegend=False)
    return fig

def create_radar_chart(text_emotions, audio_emotions, face_emotions, fused_emotions):
    """Create a radar chart comparing all modalities"""
    emotions = list(text_emotions.keys())
    
    fig = go.Figure()
    
    # Add traces for each modality
    if text_emotions:
        fig.add_trace(go.Scatterpolar(
            r=list(text_emotions.values()),
            theta=emotions,
            fill='toself',
            name='Text',
            line_color='blue'
        ))
    
    if audio_emotions:
        fig.add_trace(go.Scatterpolar(
            r=list(audio_emotions.values()),
            theta=emotions,
            fill='toself',
            name='Audio',
            line_color='red'
        ))
    
    if face_emotions:
        fig.add_trace(go.Scatterpolar(
            r=list(face_emotions.values()),
            theta=emotions,
            fill='toself',
            name='Face',
            line_color='green'
        ))
    
    if fused_emotions:
        fig.add_trace(go.Scatterpolar(
            r=list(fused_emotions.values()),
            theta=emotions,
            fill='toself',
            name='Fused',
            line_color='purple',
            line_width=3
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Multimodal Emotion Comparison"
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="Multimodal Emotion Recognition",
        page_icon="😊",
        layout="wide"
    )
    
    st.title("🎭 Multimodal Emotion Recognition System")
    st.markdown("Analyze emotions from text, audio, and facial expressions using AI")
    
    # Load analyzers
    try:
        text_analyzer, audio_analyzer, face_analyzer, fusion = load_analyzers()
        
        # Show a note about available features
        if not hasattr(text_analyzer, 'classifier') or text_analyzer.classifier is None:
            st.sidebar.info("📝 Text analysis: Using advanced rule-based detection")
        else:
            st.sidebar.success("📝 Text analysis: AI model loaded")
            
        st.sidebar.info("🎵 Audio analysis: Using file-based feature extraction")
        st.sidebar.info("😊 Face analysis: Using OpenCV detection")
        
    except Exception as e:
        st.error(f"Error loading analyzers: {str(e)}")
        st.stop()
    
    # Sidebar for input selection
    st.sidebar.title("Input Options")
    analysis_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["Single Modality", "Multimodal Analysis"]
    )
    
    # Initialize session state for storing results
    if 'text_emotions' not in st.session_state:
        st.session_state.text_emotions = {}
    if 'audio_emotions' not in st.session_state:
        st.session_state.audio_emotions = {}
    if 'face_emotions' not in st.session_state:
        st.session_state.face_emotions = {}
    if 'fused_emotions' not in st.session_state:
        st.session_state.fused_emotions = {}
    
    if analysis_mode == "Single Modality":
        modality = st.sidebar.selectbox(
            "Select Modality",
            ["Text", "Audio", "Facial Expression"]
        )
        
        if modality == "Text":
            st.header("📝 Text Emotion Analysis")
            
            text_input_method = st.radio(
                "Choose input method:",
                ["Type text", "Upload file"]
            )
            
            text_content = ""
            if text_input_method == "Type text":
                text_content = st.text_area("Enter text to analyze:", height=150)
            else:
                uploaded_file = st.file_uploader(
                    "Upload a text file",
                    type=['txt'],
                    help="Upload a .txt file containing the text to analyze"
                )
                if uploaded_file:
                    text_content = str(uploaded_file.read(), "utf-8")
                    st.text_area("File content:", text_content, height=150, disabled=True)
            
            if st.button("Analyze Text Emotion") and text_content.strip():
                with st.spinner("Analyzing text emotions..."):
                    try:
                        emotions = text_analyzer.analyze(text_content)
                        st.session_state.text_emotions = emotions
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Emotion Scores")
                            for emotion, score in emotions.items():
                                st.metric(emotion.capitalize(), f"{score:.2%}")
                        
                        with col2:
                            fig = create_emotion_chart(emotions, "Text Emotion Analysis")
                            st.plotly_chart(fig, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error analyzing text: {str(e)}")
        
        elif modality == "Audio":
            st.header("🎵 Audio Emotion Analysis")
            
            uploaded_audio = st.file_uploader(
                "Upload an audio file",
                type=['wav', 'mp3'],
                help="Upload a .wav or .mp3 audio file for emotion analysis",
                accept_multiple_files=False
            )
            
            if uploaded_audio:
                st.success(f"✅ Audio file uploaded: {uploaded_audio.name} ({get_file_size_mb(uploaded_audio):.1f} MB)")
                
                if st.button("Analyze Audio Emotion"):
                    with st.spinner("Analyzing audio emotions..."):
                        try:
                            # Save uploaded file temporarily
                            temp_path = save_uploaded_file(uploaded_audio)
                            
                            if temp_path is None:
                                st.error("Failed to process uploaded file. Please try again.")
                                st.stop()
                            
                            emotions = audio_analyzer.analyze(temp_path)
                            st.session_state.audio_emotions = emotions
                            
                            # Display results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Emotion Scores")
                                for emotion, score in emotions.items():
                                    st.metric(emotion.capitalize(), f"{score:.2%}")
                            
                            with col2:
                                fig = create_emotion_chart(emotions, "Audio Emotion Analysis")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Clean up temp file
                            if temp_path:
                                os.unlink(temp_path)
                                
                        except Exception as e:
                            st.error(f"Error analyzing audio: {str(e)}")
        
        elif modality == "Facial Expression":
            st.header("😊 Facial Expression Analysis")
            
            face_input_method = st.radio(
                "Choose input method:",
                ["Upload image", "Use webcam"]
            )
            
            if face_input_method == "Upload image":
                uploaded_image = st.file_uploader(
                    "Upload an image",
                    type=['jpg', 'jpeg', 'png'],
                    help="Upload a .jpg, .jpeg, or .png image containing a face",
                    accept_multiple_files=False
                )
                
                if uploaded_image:
                    st.image(uploaded_image, caption="Uploaded Image", width=300)
                    st.success(f"✅ Image uploaded: {uploaded_image.name} ({get_file_size_mb(uploaded_image):.1f} MB)")
                    
                    if st.button("Analyze Facial Expression"):
                        with st.spinner("Analyzing facial expressions..."):
                            try:
                                # Save uploaded file temporarily
                                temp_path = save_uploaded_file(uploaded_image)
                                
                                if temp_path is None:
                                    st.error("Failed to process uploaded image. Please try again.")
                                    st.stop()
                                
                                emotions = face_analyzer.analyze(temp_path)
                                st.session_state.face_emotions = emotions
                                
                                # Display results
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.subheader("Emotion Scores")
                                    for emotion, score in emotions.items():
                                        st.metric(emotion.capitalize(), f"{score:.2%}")
                                
                                with col2:
                                    fig = create_emotion_chart(emotions, "Facial Expression Analysis")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Clean up temp file
                                if temp_path:
                                    os.unlink(temp_path)
                                
                            except Exception as e:
                                st.error(f"Error analyzing facial expression: {str(e)}")
            
            else:  # Webcam
                st.subheader("📷 Live Webcam Analysis")
                
                # Camera input
                camera_image = st.camera_input("Take a photo for emotion analysis")
                
                if camera_image:
                    st.image(camera_image, caption="Captured Image", width=300)
                    st.success("✅ Photo captured successfully!")
                    
                    if st.button("Analyze Facial Expression from Camera"):
                        with st.spinner("Analyzing facial expressions from camera..."):
                            try:
                                # Save camera image temporarily
                                temp_path = save_uploaded_file(camera_image)
                                
                                if temp_path is None:
                                    st.error("Failed to process camera image. Please try again.")
                                    st.stop()
                                
                                emotions = face_analyzer.analyze(temp_path)
                                st.session_state.face_emotions = emotions
                                
                                # Display results
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.subheader("Emotion Scores")
                                    for emotion, score in emotions.items():
                                        st.metric(emotion.capitalize(), f"{score:.2%}")
                                
                                with col2:
                                    fig = create_emotion_chart(emotions, "Webcam Facial Expression Analysis")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Clean up temp file
                                if temp_path:
                                    os.unlink(temp_path)
                                    
                            except Exception as e:
                                st.error(f"Error analyzing camera image: {str(e)}")
                                
                else:
                    st.info("Click the camera button above to take a photo for emotion analysis")
    
    else:  # Multimodal Analysis
        st.header("🔄 Multimodal Emotion Analysis")
        st.markdown("Analyze emotions from multiple modalities and see the fused results")
        
        # Create tabs for different inputs
        tab1, tab2, tab3 = st.tabs(["📝 Text Input", "🎵 Audio Input", "😊 Face Input"])
        
        with tab1:
            text_content = st.text_area("Enter text to analyze:", height=100)
            if st.button("Analyze Text", key="multi_text") and text_content.strip():
                with st.spinner("Analyzing text..."):
                    try:
                        emotions = text_analyzer.analyze(text_content)
                        st.session_state.text_emotions = emotions
                        st.success("Text analysis complete!")
                    except Exception as e:
                        st.error(f"Error analyzing text: {str(e)}")
        
        with tab2:
            uploaded_audio = st.file_uploader(
                "Upload audio file",
                type=['wav', 'mp3'],
                key="multi_audio",
                accept_multiple_files=False
            )
            if uploaded_audio:
                st.success(f"✅ Audio file uploaded: {uploaded_audio.name} ({get_file_size_mb(uploaded_audio):.1f} MB)")
                if st.button("Analyze Audio", key="multi_audio_btn"):
                    with st.spinner("Analyzing audio..."):
                        try:
                            temp_path = save_uploaded_file(uploaded_audio)
                            if temp_path is None:
                                st.error("Failed to process uploaded file. Please try again.")
                            else:
                                emotions = audio_analyzer.analyze(temp_path)
                                st.session_state.audio_emotions = emotions
                                st.success("Audio analysis complete!")
                                if temp_path:
                                    os.unlink(temp_path)
                        except Exception as e:
                            st.error(f"Error analyzing audio: {str(e)}")
        
        with tab3:
            face_input_method = st.radio(
                "Choose input method:",
                ["Upload image", "Use webcam"],
                key="multi_face_method"
            )
            
            if face_input_method == "Upload image":
                uploaded_image = st.file_uploader(
                    "Upload image file",
                    type=['jpg', 'jpeg', 'png'],
                    key="multi_face",
                    accept_multiple_files=False
                )
                if uploaded_image:
                    st.image(uploaded_image, width=200)
                    st.success(f"✅ Image uploaded: {uploaded_image.name} ({get_file_size_mb(uploaded_image):.1f} MB)")
                    if st.button("Analyze Face", key="multi_face_btn"):
                        with st.spinner("Analyzing facial expression..."):
                            try:
                                temp_path = save_uploaded_file(uploaded_image)
                                if temp_path is None:
                                    st.error("Failed to process uploaded image. Please try again.")
                                else:
                                    emotions = face_analyzer.analyze(temp_path)
                                    st.session_state.face_emotions = emotions
                                    st.success("Facial expression analysis complete!")
                                    if temp_path:
                                        os.unlink(temp_path)
                            except Exception as e:
                                st.error(f"Error analyzing facial expression: {str(e)}")
            else:
                camera_image = st.camera_input("Take a photo for emotion analysis", key="multi_camera")
                if camera_image:
                    st.image(camera_image, width=200)
                    st.success("✅ Photo captured successfully!")
                    if st.button("Analyze Face from Camera", key="multi_camera_btn"):
                        with st.spinner("Analyzing facial expression from camera..."):
                            try:
                                temp_path = save_uploaded_file(camera_image)
                                if temp_path is None:
                                    st.error("Failed to process camera image. Please try again.")
                                else:
                                    emotions = face_analyzer.analyze(temp_path)
                                    st.session_state.face_emotions = emotions
                                    st.success("Facial expression analysis complete!")
                                    if temp_path:
                                        os.unlink(temp_path)
                            except Exception as e:
                                st.error(f"Error analyzing camera image: {str(e)}")
                else:
                    st.info("Click the camera button above to take a photo")
        
        # Fusion and Results Section
        st.subheader("🔬 Multimodal Fusion Results")
        
        # Check if we have at least one analysis
        has_results = any([
            st.session_state.text_emotions,
            st.session_state.audio_emotions,
            st.session_state.face_emotions
        ])
        
        if has_results:
            if st.button("Perform Multimodal Fusion"):
                with st.spinner("Fusing multimodal predictions..."):
                    try:
                        fused_emotions = fusion.fuse_predictions(
                            st.session_state.text_emotions,
                            st.session_state.audio_emotions,
                            st.session_state.face_emotions
                        )
                        st.session_state.fused_emotions = fused_emotions
                        st.success("Multimodal fusion complete!")
                    except Exception as e:
                        st.error(f"Error in fusion: {str(e)}")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Individual Results")
                if st.session_state.text_emotions:
                    st.write("**Text Emotions:**")
                    for emotion, score in st.session_state.text_emotions.items():
                        st.write(f"- {emotion.capitalize()}: {score:.2%}")
                
                if st.session_state.audio_emotions:
                    st.write("**Audio Emotions:**")
                    for emotion, score in st.session_state.audio_emotions.items():
                        st.write(f"- {emotion.capitalize()}: {score:.2%}")
                
                if st.session_state.face_emotions:
                    st.write("**Face Emotions:**")
                    for emotion, score in st.session_state.face_emotions.items():
                        st.write(f"- {emotion.capitalize()}: {score:.2%}")
            
            with col2:
                if st.session_state.fused_emotions:
                    st.subheader("Fused Results")
                    for emotion, score in st.session_state.fused_emotions.items():
                        st.metric(emotion.capitalize(), f"{score:.2%}")
            
            # Visualization
            if any([st.session_state.text_emotions, st.session_state.audio_emotions, 
                   st.session_state.face_emotions, st.session_state.fused_emotions]):
                
                st.subheader("📊 Emotion Comparison")
                radar_fig = create_radar_chart(
                    st.session_state.text_emotions,
                    st.session_state.audio_emotions,
                    st.session_state.face_emotions,
                    st.session_state.fused_emotions
                )
                st.plotly_chart(radar_fig, use_container_width=True)
        
        else:
            st.info("Please analyze at least one modality to see fusion results.")
        
        # Clear results button
        if st.button("Clear All Results"):
            st.session_state.text_emotions = {}
            st.session_state.audio_emotions = {}
            st.session_state.face_emotions = {}
            st.session_state.fused_emotions = {}
            st.rerun()

if __name__ == "__main__":
    main()
