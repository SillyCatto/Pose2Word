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
from model.sign_classifier import create_model
from model.trainer import train_model
from model.dataset import SignLanguageDataset
from model.raft_flow_extractor import RAFTFlowExtractor


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
        col1, col2 = st.columns(2)
        
        with col1:
            landmarks_dir = st.text_input(
                "Landmarks Directory",
                value="extracted_landmarks",
                help="Directory containing .npy landmark files organized by class"
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
    Test your trained model on individual samples.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_path = st.text_input(
            "Model Path",
            value="checkpoints/lstm_run1/best_model.pth"
        )
    
    with col2:
        sample_path = st.file_uploader(
            "Upload Sample (.npy)",
            type=["npy"],
            help="Upload a .npy file with landmark features"
        )
    
    if sample_path and st.button("🎯 Predict", type="primary"):
        st.info("Inference feature coming soon!")
        st.write("This will:")
        st.write("- Load the sample")
        st.write("- Extract flow features")
        st.write("- Run model prediction")
        st.write("- Show top-k predictions with confidence scores")


if __name__ == "__main__":
    # For testing the page standalone
    render()
