"""
Flow Extraction Page - Extract RAFT Optical Flow from Videos
"""

import streamlit as st
from pathlib import Path
import sys
import cv2
import numpy as np
import tempfile

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "model"))

from raft_flow_extractor import RAFTFlowExtractor


def render():
    """Render the RAFT flow extraction page."""
    st.header("🌊 RAFT Optical Flow Extractor")
    
    st.markdown("""
    Extract optical flow features from videos using pre-trained RAFT models.
    This captures motion information between frames for sign language recognition.
    """)
    
    # Configuration
    with st.expander("⚙️ Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_type = st.selectbox(
                "RAFT Model",
                options=["small", "large"],
                help="Small: Faster, less accurate | Large: Slower, more accurate"
            )
        
        with col2:
            feature_dim = st.number_input(
                "Feature Dimension",
                min_value=16,
                max_value=256,
                value=32,
                step=16,
                help="Dimension of pooled flow features"
            )
        
        with col3:
            device = st.selectbox(
                "Device",
                options=["auto", "mps", "cuda", "cpu"],
                help="Processing device"
            )
    
    # File input
    st.markdown("---")
    st.subheader("📁 Input")
    
    input_mode = st.radio(
        "Input Mode",
        options=["Upload Video", "Select from Directory"],
        horizontal=True
    )
    
    video_path = None
    
    if input_mode == "Upload Video":
        uploaded_file = st.file_uploader(
            "Upload Video File",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload a sign language video"
        )
        
        if uploaded_file:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            st.success(f"✓ Uploaded: {uploaded_file.name}")
            
    else:  # Select from Directory
        video_dir = st.text_input(
            "Video Directory",
            value="videos",
            help="Directory containing video files"
        )
        
        if video_dir and Path(video_dir).exists():
            video_files = list(Path(video_dir).glob("*.mp4")) + \
                         list(Path(video_dir).glob("*.avi")) + \
                         list(Path(video_dir).glob("*.mov"))
            
            if video_files:
                selected_video = st.selectbox(
                    "Select Video",
                    options=[v.name for v in video_files]
                )
                video_path = str(Path(video_dir) / selected_video)
                st.success(f"✓ Selected: {selected_video}")
            else:
                st.warning("No video files found in directory")
    
    # Output configuration
    st.markdown("---")
    st.subheader("💾 Output")
    
    col1, col2 = st.columns(2)
    
    with col1:
        output_dir = st.text_input(
            "Output Directory",
            value="extracted_flow",
            help="Directory to save flow features"
        )
    
    with col2:
        output_name = st.text_input(
            "Output Filename",
            value="flow_features.npy",
            help="Name for the output .npy file"
        )
    
    # Extract button
    st.markdown("---")
    
    if st.button("🚀 Extract Flow Features", type="primary", disabled=video_path is None):
        if not video_path:
            st.error("Please select or upload a video first!")
            return
        
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            with st.spinner("Initializing RAFT model..."):
                # Initialize extractor
                device_type = device if device != "auto" else None
                extractor = RAFTFlowExtractor(
                    model_size=model_type,
                    device=device_type
                )
            
            st.info(f"Using device: **{extractor.device}**")
            
            # Read video
            with st.spinner("Reading video frames..."):
                cap = cv2.VideoCapture(video_path)
                frames = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Convert BGR to RGB
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                cap.release()
            
            st.success(f"✓ Read {len(frames)} frames")
            
            if len(frames) < 2:
                st.error("Video must have at least 2 frames for optical flow!")
                return
            
            # Extract flow
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Extracting optical flow...")
            
            # Extract flow features
            flow_features = extractor.extract_global_flow_features(
                frames=frames,
                feature_dim=feature_dim
            )
            
            progress_bar.progress(1.0)
            status_text.text("Flow extraction complete!")
            
            # Save features
            output_file = output_path / output_name
            np.save(output_file, flow_features)
            
            st.success(f"✅ Saved flow features to: {output_file}")
            
            # Show feature info
            st.markdown("---")
            st.subheader("📊 Feature Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Number of Frames", len(frames))
            
            with col2:
                st.metric("Flow Features Shape", f"{flow_features.shape}")
            
            with col3:
                st.metric("File Size", f"{output_file.stat().st_size / 1024:.2f} KB")
            
            # Show sample flow statistics
            st.markdown("**Feature Statistics:**")
            st.write(f"- Mean: {flow_features.mean():.4f}")
            st.write(f"- Std: {flow_features.std():.4f}")
            st.write(f"- Min: {flow_features.min():.4f}")
            st.write(f"- Max: {flow_features.max():.4f}")
            
        except Exception as e:
            st.error(f"❌ Flow extraction failed: {str(e)}")
            st.exception(e)
    
    # Batch processing
    st.markdown("---")
    st.subheader("📦 Batch Processing")
    
    st.markdown("""
    Process multiple videos at once organized by class.
    
    **Expected structure:**
    ```
    videos/
    ├── class1/
    │   ├── video1.mp4
    │   └── video2.mp4
    ├── class2/
    │   └── video1.mp4
    ```
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        batch_input_dir = st.text_input(
            "Input Directory",
            value="videos",
            help="Directory with class subdirectories"
        )
    
    with col2:
        batch_output_dir = st.text_input(
            "Output Directory",
            value="extracted_flow",
            help="Directory to save flow features (maintains class structure)"
        )
    
    if st.button("🔄 Batch Process All Videos"):
        if not Path(batch_input_dir).exists():
            st.error(f"Input directory not found: {batch_input_dir}")
            return
        
        try:
            # Initialize extractor
            with st.spinner("Initializing RAFT model..."):
                device_type = device if device != "auto" else None
                extractor = RAFTFlowExtractor(
                    model_size=model_type,
                    device=device_type
                )
            
            st.info(f"Using device: **{extractor.device}**")
            
            # Find all videos
            input_path = Path(batch_input_dir)
            class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
            
            if not class_dirs:
                st.warning("No class subdirectories found!")
                return
            
            total_videos = 0
            for class_dir in class_dirs:
                videos = list(class_dir.glob("*.mp4")) + \
                        list(class_dir.glob("*.avi")) + \
                        list(class_dir.glob("*.mov"))
                total_videos += len(videos)
            
            st.write(f"Found {total_videos} videos across {len(class_dirs)} classes")
            
            # Process each class
            progress_bar = st.progress(0)
            status_text = st.empty()
            processed = 0
            
            for class_dir in class_dirs:
                class_name = class_dir.name
                output_class_dir = Path(batch_output_dir) / class_name
                output_class_dir.mkdir(parents=True, exist_ok=True)
                
                videos = list(class_dir.glob("*.mp4")) + \
                        list(class_dir.glob("*.avi")) + \
                        list(class_dir.glob("*.mov"))
                
                for video_file in videos:
                    status_text.text(f"Processing: {class_name}/{video_file.name}")
                    
                    try:
                        # Read video
                        cap = cv2.VideoCapture(str(video_file))
                        frames = []
                        
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        
                        cap.release()
                        
                        if len(frames) < 2:
                            st.warning(f"Skipping {video_file.name}: less than 2 frames")
                            continue
                        
                        # Extract flow
                        flow_features = extractor.extract_global_flow_features(
                            frames=frames,
                            feature_dim=feature_dim
                        )
                        
                        # Save with same name
                        output_file = output_class_dir / f"{video_file.stem}.npy"
                        np.save(output_file, flow_features)
                        
                        processed += 1
                        progress_bar.progress(processed / total_videos)
                        
                    except Exception as e:
                        st.warning(f"Failed to process {video_file.name}: {str(e)}")
            
            status_text.text("Batch processing complete!")
            st.success(f"✅ Processed {processed}/{total_videos} videos")
            
        except Exception as e:
            st.error(f"❌ Batch processing failed: {str(e)}")
            st.exception(e)
