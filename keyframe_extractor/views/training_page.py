"""
Training & Testing Page for Streamlit App

UI for training and testing sign language recognition models.
"""

import streamlit as st
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional
import json

# Add model directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "model"))

# Import model components
from sign_classifier import create_model
from trainer import train_model
from dataset import SignLanguageDataset
from raft_flow_extractor import RAFTFlowExtractor


def render():
    """Render the training & testing page."""
    st.header("🤖 Model Training & Testing")
    
    # Create tabs for different operations
    tab_train, tab_test, tab_inference = st.tabs([
        "🎯 Train Model", 
        "📊 Test & Evaluate", 
        "🔮 Inference"
    ])
    
    with tab_train:
        render_training_tab()
    
    with tab_test:
        render_testing_tab()
    
    with tab_inference:
        render_inference_tab()


def render_training_tab():
    """Render the training configuration and execution tab."""
    st.subheader("Train Sign Language Recognition Model")
    
    st.markdown("""
    Train a model that combines:
    - **MediaPipe Landmarks**: Spatial features (pose + hands)
    - **RAFT Optical Flow**: Temporal motion features
    """)
    
    # Configuration
    with st.expander("📁 Data Configuration", expanded=True):
        st.info("""
        **Expected Directory Structure:**
        ```
        landmarks_directory/
        ├── class1/
        │   ├── sample1.npy
        │   ├── sample2.npy
        │   └── ...
        ├── class2/
        │   ├── sample1.npy
        │   └── ...
        ```
        Each class should have its own subdirectory with .npy files.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            landmarks_dir = st.text_input(
                "Landmarks Directory",
                value="extracted_landmarks",
                help="Path to directory containing subdirectories for each class"
            )
        
        with col2:
            flow_dir = st.text_input(
                "Flow Directory (Optional)",
                value="",
                help="Directory with pre-extracted flow features. Leave empty to use landmarks only."
            )
        
        # Check if directories exist
        if landmarks_dir and Path(landmarks_dir).exists():
            st.success(f"✓ Found landmarks directory: {landmarks_dir}")
            
            # Show dataset info
            try:
                dataset = SignLanguageDataset(landmarks_dir=Path(landmarks_dir))
                dist = dataset.get_class_distribution()
                
                st.write(f"**Dataset:** {len(dataset)} samples from {len(dataset.classes)} classes")
                
                # Show distribution
                df = pd.DataFrame(list(dist.items()), columns=["Class", "Samples"])
                fig = px.bar(df, x="Class", y="Samples", title="Class Distribution")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load dataset: {e}")
        else:
            st.error(f"❌ Landmarks directory not found: {landmarks_dir}")
    
    # Model configuration
    with st.expander("🧠 Model Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_type = st.selectbox(
                "Model Type",
                options=["lstm", "transformer", "hybrid"],
                help="LSTM: Fast, good for sequential data\nTransformer: Better accuracy, slower\nHybrid: Best of both"
            )
        
        with col2:
            num_classes = st.number_input(
                "Number of Classes",
                min_value=2,
                max_value=100,
                value=15,
                help="Number of sign language classes"
            )
        
        with col3:
            hidden_dim = st.number_input(
                "Hidden Dimension",
                min_value=64,
                max_value=512,
                value=256,
                step=64,
                help="Model hidden layer size"
            )
    
    # Training configuration
    with st.expander("⚙️ Training Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=128,
                value=16,
                help="Number of samples per batch"
            )
            
            num_epochs = st.number_input(
                "Number of Epochs",
                min_value=1,
                max_value=500,
                value=50,
                help="Training epochs"
            )
        
        with col2:
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=0.00001,
                max_value=0.1,
                value=0.001,
                format="%.5f",
                help="Initial learning rate"
            )
            
            weight_decay = st.number_input(
                "Weight Decay",
                min_value=0.0,
                max_value=0.01,
                value=0.0001,
                format="%.5f",
                help="L2 regularization"
            )
        
        with col3:
            device = st.selectbox(
                "Device",
                options=["auto", "mps", "cuda", "cpu"],
                help="Training device. 'auto' detects MPS/CUDA automatically"
            )
            
            use_amp = st.checkbox(
                "Mixed Precision",
                value=True,
                help="Use automatic mixed precision for faster training"
            )
    
    # Save configuration
    with st.expander("💾 Save Configuration"):
        save_dir = st.text_input(
            "Checkpoint Directory",
            value=f"checkpoints/{model_type}_run1",
            help="Directory to save model checkpoints"
        )
        
        experiment_name = st.text_input(
            "Experiment Name",
            value="sign_language_experiment",
            help="Name for TensorBoard logging"
        )
    
    # Training controls
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        train_button = st.button(
            "🚀 Start Training",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        if st.button("💾 Save Config", use_container_width=True):
            config = {
                "landmarks_dir": landmarks_dir,
                "flow_dir": flow_dir if flow_dir else None,
                "model_type": model_type,
                "num_classes": num_classes,
                "hidden_dim": hidden_dim,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "device": device if device != "auto" else None,
                "save_dir": save_dir
            }
            
            config_path = Path(save_dir) / "config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            st.success(f"✓ Configuration saved to {config_path}")
    
    with col3:
        if st.button("📋 Load Config", use_container_width=True):
            st.info("Config loading coming soon!")
    
    # Execute training
    if train_button:
        if not Path(landmarks_dir).exists():
            st.error("❌ Landmarks directory not found!")
            return
        
        st.markdown("---")
        st.subheader("🎯 Training Progress")
        
        # Show device info
        device_type = device if device != "auto" else None
        if device_type is None:
            if torch.backends.mps.is_available():
                device_type = "mps"
            elif torch.cuda.is_available():
                device_type = "cuda"
            else:
                device_type = "cpu"
        
        st.info(f"Training on: **{device_type.upper()}**")
        
        # Training placeholder
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_placeholder = st.empty()
        
        try:
            # Start training
            with st.spinner("Initializing training..."):
                status_text.text("Loading data and creating model...")
                
                # Call training function
                train_model(
                    landmarks_dir=landmarks_dir,
                    flow_dir=flow_dir if flow_dir else None,
                    model_type=model_type,
                    num_classes=num_classes,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    save_dir=save_dir,
                    device=device_type
                )
            
            st.success("✅ Training completed successfully!")
            st.balloons()
            
            # Show results
            history_path = Path(save_dir) / "training_history.json"
            if history_path.exists():
                with open(history_path) as f:
                    history = json.load(f)
                
                # Plot training curves
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history["train_loss"],
                    name="Train Loss",
                    mode='lines'
                ))
                fig.add_trace(go.Scatter(
                    y=history["val_loss"],
                    name="Val Loss",
                    mode='lines'
                ))
                fig.update_layout(title="Training Loss", xaxis_title="Epoch", yaxis_title="Loss")
                st.plotly_chart(fig, use_container_width=True)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    y=history["train_acc"],
                    name="Train Accuracy",
                    mode='lines'
                ))
                fig2.add_trace(go.Scatter(
                    y=history["val_acc"],
                    name="Val Accuracy",
                    mode='lines'
                ))
                fig2.update_layout(title="Training Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy")
                st.plotly_chart(fig2, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            st.exception(e)


def render_testing_tab():
    """Render the model testing and evaluation tab."""
    st.subheader("Test & Evaluate Model")
    
    st.markdown("""
    Load a trained model and evaluate it on test data.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        checkpoint_path = st.text_input(
            "Model Checkpoint",
            value="checkpoints/lstm_run1/best_model.pth",
            help="Path to saved model checkpoint"
        )
    
    with col2:
        test_data_dir = st.text_input(
            "Test Data Directory",
            value="test_landmarks",
            help="Directory with test landmark files"
        )
    
    if st.button("🧪 Run Evaluation", type="primary"):
        st.info("Evaluation feature coming soon!")
        st.write("This will:")
        st.write("- Load the trained model")
        st.write("- Run inference on test set")
        st.write("- Show confusion matrix")
        st.write("- Display per-class metrics")


def render_inference_tab():
    """Render the single-sample inference tab."""
    st.subheader("🔮 Single Sample Inference")
    
    st.markdown("""
    Test your trained model on individual videos or landmark files.
    """)
    
    # Model configuration
    with st.expander("🧠 Model Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            checkpoint_path = st.text_input(
                "Model Checkpoint",
                value="checkpoints/lstm_run1/best_model.pth",
                help="Path to trained model checkpoint"
            )
        
        with col2:
            class_names_file = st.text_input(
                "Class Names File (Optional)",
                value="",
                help="JSON file with class names. If empty, will use class indices."
            )
    
    # Input mode
    st.markdown("---")
    st.subheader("📁 Input")
    
    input_mode = st.radio(
        "Input Type",
        options=["Video File", "Landmark File (.npy)"],
        horizontal=True
    )
    
    input_data = None
    is_video = False
    
    if input_mode == "Video File":
        uploaded_video = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload a sign language video for inference"
        )
        
        if uploaded_video:
            # Save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tmp_file:
                tmp_file.write(uploaded_video.read())
                input_data = tmp_file.name
            is_video = True
            st.success(f"✓ Uploaded: {uploaded_video.name}")
    
    else:  # Landmark File
        uploaded_landmarks = st.file_uploader(
            "Upload Landmarks (.npy)",
            type=["npy"],
            help="Upload pre-extracted landmark features"
        )
        
        if uploaded_landmarks:
            # Save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp_file:
                tmp_file.write(uploaded_landmarks.read())
                input_data = tmp_file.name
            is_video = False
            st.success(f"✓ Uploaded: {uploaded_landmarks.name}")
    
    # Processing options
    with st.expander("⚙️ Processing Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            use_flow = st.checkbox(
                "Extract Flow Features",
                value=True,
                help="Extract RAFT optical flow (only for video input)"
            )
            
            if use_flow:
                flow_model = st.selectbox(
                    "RAFT Model",
                    options=["small", "large"],
                    help="RAFT model size"
                )
        
        with col2:
            top_k = st.number_input(
                "Top-K Predictions",
                min_value=1,
                max_value=10,
                value=3,
                help="Show top K predictions"
            )
            
            device = st.selectbox(
                "Device",
                options=["auto", "mps", "cuda", "cpu"]
            )
    
    # Predict button
    st.markdown("---")
    
    if st.button("🎯 Predict", type="primary", disabled=input_data is None):
        if not input_data:
            st.error("Please upload a file first!")
            return
        
        if not Path(checkpoint_path).exists():
            st.error(f"Model checkpoint not found: {checkpoint_path}")
            return
        
        try:
            import torch
            import cv2
            
            # Auto-detect device
            if device == "auto":
                if torch.backends.mps.is_available():
                    device_type = "mps"
                elif torch.cuda.is_available():
                    device_type = "cuda"
                else:
                    device_type = "cpu"
            else:
                device_type = device
            
            device_obj = torch.device(device_type)
            st.info(f"Using device: **{device_type.upper()}**")
            
            # Load checkpoint to get model config
            with st.spinner("Loading model..."):
                checkpoint = torch.load(checkpoint_path, map_location=device_obj)
                model_config = checkpoint.get("config", {})
                
                # Get model architecture
                model_type = model_config.get("model_type", "lstm")
                num_classes = model_config.get("num_classes", 15)
                hidden_dim = model_config.get("hidden_dim", 256)
                
                st.write(f"**Model:** {model_type.upper()} | **Classes:** {num_classes}")
            
            # Process input
            landmarks = None
            flow_features = None
            
            if is_video:
                # Extract landmarks from video
                with st.spinner("Extracting landmarks from video..."):
                    from landmark_extractor import extract_landmarks_from_frames
                    from data_exporter import landmarks_to_numpy
                    from video_utils import get_frames_from_video
                    
                    video_frames = get_frames_from_video(input_data)
                    if not video_frames:
                        st.error("Failed to read video frames!")
                        return
                    
                    raw_landmarks = extract_landmarks_from_frames(
                        video_frames, method="MediaPipe Pose + Hands"
                    )
                    landmarks = landmarks_to_numpy(raw_landmarks, num_frames=30)
                    
                    if landmarks is None or len(landmarks) == 0:
                        st.error("Failed to extract landmarks from video!")
                        return
                    
                    st.success(f"✓ Extracted landmarks: {landmarks.shape}")
                
                # Extract flow if requested
                if use_flow:
                    with st.spinner("Extracting RAFT optical flow..."):
                        from raft_flow_extractor import RAFTFlowExtractor
                        
                        # Read video frames
                        cap = cv2.VideoCapture(input_data)
                        frames = []
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        cap.release()
                        
                        # Extract flow
                        extractor = RAFTFlowExtractor(model_size=flow_model, device=device_type)
                        flows = extractor.extract_flow_from_frames(frames, return_magnitude=True)
                        flow_features = extractor.extract_global_flow_features(flows, pool_size=(4, 4))
                        
                        st.success(f"✓ Extracted flow: {flow_features.shape}")
            
            else:
                # Load landmarks from file
                with st.spinner("Loading landmarks..."):
                    landmarks = np.load(input_data)
                    st.success(f"✓ Loaded landmarks: {landmarks.shape}")
            
            # Create model
            with st.spinner("Initializing model..."):
                from sign_classifier import LSTMSignClassifier, TransformerSignClassifier, HybridSignClassifier
                
                if model_type == "lstm":
                    model = LSTMSignClassifier(
                        num_classes=num_classes,
                        hidden_dim=hidden_dim,
                    )
                elif model_type == "transformer":
                    model = TransformerSignClassifier(
                        num_classes=num_classes,
                        d_model=hidden_dim,
                    )
                else:  # hybrid
                    model = HybridSignClassifier(
                        num_classes=num_classes,
                        hidden_dim=hidden_dim,
                    )
                
                # Load weights
                model.load_state_dict(checkpoint["model_state_dict"])
                model = model.to(device_obj)
                model.eval()
            
            # Run inference
            with st.spinner("Running inference..."):
                with torch.no_grad():
                    # Prepare input
                    landmarks_tensor = torch.from_numpy(landmarks).float().unsqueeze(0).to(device_obj)
                    
                    if flow_features is not None:
                        flow_tensor = torch.from_numpy(flow_features).float().unsqueeze(0).to(device_obj)
                    else:
                        # Models require flow input; provide zeros
                        flow_tensor = torch.zeros(1, landmarks.shape[0] - 1, 32).to(device_obj)
                    
                    logits = model(landmarks_tensor, flow_tensor)
                    
                    # Get probabilities
                    probs = torch.softmax(logits, dim=-1)
                    top_probs, top_indices = torch.topk(probs, k=min(top_k, num_classes), dim=-1)
                    
                    top_probs = top_probs[0].cpu().numpy()
                    top_indices = top_indices[0].cpu().numpy()
            
            # Load class names if provided
            class_names = None
            if class_names_file and Path(class_names_file).exists():
                import json
                with open(class_names_file) as f:
                    class_names = json.load(f)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Top prediction
            top_class = class_names[top_indices[0]] if class_names else f"Class {top_indices[0]}"
            top_confidence = top_probs[0] * 100
            
            st.markdown(f"""
            ### 🏆 Top Prediction
            **{top_class}** - {top_confidence:.2f}% confidence
            """)
            
            # Show all top-k predictions
            st.markdown(f"### Top {top_k} Predictions")
            
            for i, (idx, prob) in enumerate(zip(top_indices, top_probs), 1):
                class_name = class_names[idx] if class_names else f"Class {idx}"
                confidence = prob * 100
                
                # Progress bar for confidence
                st.write(f"**{i}. {class_name}**")
                st.progress(float(prob))
                st.write(f"{confidence:.2f}%")
            
            # Feature info
            st.markdown("---")
            st.subheader("ℹ️ Input Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Landmarks:**")
                st.write(f"- Shape: {landmarks.shape}")
                st.write(f"- Frames: {landmarks.shape[0]}")
                st.write(f"- Features: {landmarks.shape[1]}")
            
            with col2:
                if flow_features is not None:
                    st.write("**Flow Features:**")
                    st.write(f"- Shape: {flow_features.shape}")
                    st.write(f"- Frames: {flow_features.shape[0]}")
                    st.write(f"- Features: {flow_features.shape[1]}")
                else:
                    st.write("**Flow Features:** Not used")
        
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    # For testing the page standalone
    render()
