# Sign Language Recognition Model

Deep learning models combining RAFT optical flow and MediaPipe landmarks for sign language recognition.

## Architecture

### Three Model Types

1. **LSTM Model** (`lstm`)
   - Fast training and inference
   - Good for sequential temporal data
   - Best for: Real-time applications, limited compute

2. **Transformer Model** (`transformer`)
   - Self-attention mechanism
   - Better accuracy on complex patterns
   - Best for: Offline processing, maximum accuracy

3. **Hybrid Model** (`hybrid`)
   - Combines LSTM encoding with attention
   - Balance between speed and accuracy
   - Best for: Production deployments

## Features

### Input Streams

- **Spatial Features**: MediaPipe landmarks (258 dimensions)
  - Body pose: 33 landmarks × 4 (x, y, z, visibility) = 132 features
  - Left hand: 21 landmarks × 3 (x, y, z) = 63 features
  - Right hand: 21 landmarks × 3 (x, y, z) = 63 features

- **Temporal Features**: RAFT optical flow
  - Motion magnitude and direction
  - Spatial pooling for global features
  - Captures hand/body movement patterns

### M4 MacBook Optimizations

- MPS (Metal Performance Shaders) acceleration
- Mixed precision training (FP16/FP32)
- Efficient data loading with multiple workers
- Optimized batch sizes for unified memory

## Quick Start

### 1. Extract Optical Flow

```python
from model.raft_flow_extractor import RAFTFlowExtractor

# Initialize extractor
extractor = RAFTFlowExtractor(model_size="small", device="mps")

# Extract from frames
flows = extractor.extract_flow_from_frames(frames)

# Or from directory
flows = extractor.extract_flow_from_directory(
    frames_dir="keyframes/hello",
    output_dir="flow_viz/hello"
)
```

### 2. Create Dataset

```python
from model.dataset import create_data_loaders

train_loader, val_loader = create_data_loaders(
    landmarks_dir="extracted_landmarks",
    flow_dir="extracted_flows",  # Optional
    batch_size=16,
    train_split=0.8
)
```

### 3. Train Model

```python
from model.trainer import train_model

train_model(
    landmarks_dir="extracted_landmarks",
    model_type="lstm",  # or "transformer", "hybrid"
    num_classes=15,
    batch_size=16,
    num_epochs=50,
    learning_rate=0.001,
    save_dir="checkpoints/my_model"
)
```

### 4. Load and Inference

```python
import torch
from model.sign_classifier import create_model

# Create model
model = create_model("lstm", num_classes=15, device="mps")

# Load checkpoint
checkpoint = torch.load("checkpoints/my_model/best_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Inference
with torch.no_grad():
    logits = model(landmarks, flow)
    predictions = torch.argmax(logits, dim=1)
```

## File Structure

```
model/
├── __init__.py                 # Package initialization
├── raft_flow_extractor.py      # RAFT optical flow extraction
├── dataset.py                  # Dataset loader
├── sign_classifier.py          # Model architectures
├── trainer.py                  # Training logic
└── README.md                   # This file
```

## Training Configuration

### Recommended Settings

**For quick experiments (M4 MacBook):**
```python
{
    "model_type": "lstm",
    "batch_size": 16,
    "num_epochs": 30,
    "learning_rate": 0.001,
    "hidden_dim": 256
}
```

**For best accuracy:**
```python
{
    "model_type": "transformer",
    "batch_size": 8,
    "num_epochs": 100,
    "learning_rate": 0.0005,
    "d_model": 512,
    "nhead": 8,
    "num_layers": 6
}
```

**For production deployment:**
```python
{
    "model_type": "hybrid",
    "batch_size": 12,
    "num_epochs": 50,
    "learning_rate": 0.0008,
    "hidden_dim": 256
}
```

## Performance Tips

### 1. Batch Size Optimization
- Start with batch size 16
- Monitor GPU memory in Activity Monitor
- Increase if memory usage < 80%
- Decrease if you get OOM errors

### 2. Mixed Precision Training
```python
# Automatically enabled in trainer
trainer = Trainer(..., use_mixed_precision=True)
```

### 3. Learning Rate Scheduling
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=num_epochs
)
```

### 4. Data Augmentation
- Time masking for landmarks
- Flow field perturbation
- Random temporal cropping

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir checkpoints/my_model/logs
```

### Training History
```python
import json

with open("checkpoints/my_model/training_history.json") as f:
    history = json.load(f)

print(f"Best validation accuracy: {max(history['val_acc'])}")
```

## Model Selection Guide

| Use Case | Model Type | Batch Size | Notes |
|----------|-----------|------------|-------|
| Research/Experiments | LSTM | 16-32 | Fast iteration |
| Competition/Publication | Transformer | 8-16 | Best accuracy |
| Mobile/Edge Deployment | LSTM | 32 | Smallest model |
| Production API | Hybrid | 12-24 | Good balance |
| Real-time Demo | LSTM | 1-4 | Low latency |

## Troubleshooting

### OOM (Out of Memory) Errors
- Reduce batch size
- Use smaller model (LSTM instead of Transformer)
- Reduce hidden_dim or d_model
- Clear MPS cache: `torch.mps.empty_cache()`

### Slow Training
- Enable mixed precision
- Increase num_workers in DataLoader
- Use RAFT "small" instead of "large"
- Pre-extract flow features offline

### Poor Accuracy
- Train longer (more epochs)
- Increase model capacity
- Check data quality and distribution
- Add data augmentation
- Use learning rate warmup

## Citation

If you use this code, please cite:

```bibtex
@software{wlasl_sign_recognition,
  title={Sign Language Recognition with RAFT and MediaPipe},
  author={Your Team},
  year={2026},
  url={https://github.com/yourusername/sem-5-ml-project}
}
```

## References

- RAFT: [Optical Flow Estimation using Recurrent All-Pairs Field Transforms](https://arxiv.org/abs/2003.12039)
- MediaPipe: [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic)
- WLASL: [Word-Level American Sign Language Dataset](https://dxli94.github.io/WLASL/)
