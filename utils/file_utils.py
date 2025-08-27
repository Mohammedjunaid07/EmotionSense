import streamlit as st
import tempfile
import os
from pathlib import Path

def validate_file_type(uploaded_file, allowed_types):
    """
    Validate if uploaded file has allowed type
    
    Args:
        uploaded_file: Streamlit uploaded file object
        allowed_types (list): List of allowed file extensions
        
    Returns:
        bool: True if file type is allowed
    """
    if uploaded_file is None:
        return False
    
    file_extension = Path(uploaded_file.name).suffix.lower()
    return file_extension in [f".{ext}" for ext in allowed_types]

def save_uploaded_file(uploaded_file):
    """
    Save uploaded file to temporary location
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        str: Path to saved temporary file
    """
    if uploaded_file is None:
        st.error("No file provided")
        return None
        
    try:
        # Validate file size (max 50MB)
        file_size_mb = get_file_size_mb(uploaded_file)
        if file_size_mb > 50:
            st.error(f"File too large: {file_size_mb:.1f}MB. Maximum size is 50MB.")
            return None
        
        # Create a temporary directory if it doesn't exist
        temp_dir = tempfile.gettempdir()
        
        # Validate file name
        if not uploaded_file.name:
            st.error("File must have a valid name")
            return None
            
        # Create temporary file with proper suffix
        suffix = Path(uploaded_file.name).suffix.lower()
        if not suffix:
            suffix = '.tmp'
            
        # Create unique filename to avoid conflicts
        import time
        unique_name = f"upload_{int(time.time() * 1000)}{suffix}"
        temp_path = os.path.join(temp_dir, unique_name)
        
        # Write uploaded file content to temporary file
        file_bytes = uploaded_file.getvalue()
        if not file_bytes:
            st.error("File appears to be empty")
            return None
            
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(file_bytes)
            temp_file.flush()
            os.fsync(temp_file.fileno())  # Force write to disk
        
        # Verify file was written successfully
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            st.error("Failed to save file properly")
            return None
            
        return temp_path
        
    except PermissionError:
        st.error("Permission denied: Cannot save file to temporary directory")
        return None
    except OSError as e:
        st.error(f"File system error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error saving file: {str(e)}")
        return None

def cleanup_temp_file(file_path):
    """
    Clean up temporary file
    
    Args:
        file_path (str): Path to temporary file
    """
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        st.warning(f"Could not clean up temporary file: {str(e)}")

def get_file_size_mb(uploaded_file):
    """
    Get file size in MB
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        float: File size in MB
    """
    if uploaded_file is None:
        return 0.0
    
    return len(uploaded_file.getvalue()) / (1024 * 1024)

def validate_file_size(uploaded_file, max_size_mb=10):
    """
    Validate file size
    
    Args:
        uploaded_file: Streamlit uploaded file object
        max_size_mb (float): Maximum allowed file size in MB
        
    Returns:
        bool: True if file size is within limit
    """
    if uploaded_file is None:
        return False
    
    file_size = get_file_size_mb(uploaded_file)
    return file_size <= max_size_mb
