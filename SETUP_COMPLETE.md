# ✅ M4 MacBook Setup Complete for WLASL Sign Language Recognition

## Installation Summary

All prerequisites have been successfully installed and configured for training PyTorch models with RAFT on your M4 MacBook.

---

## 🎯 What Was Installed

### Core Deep Learning Framework
- **PyTorch 2.10.0** - Latest version with MPS (Metal Performance Shaders) support
- **TorchVision 0.25.0** - Computer vision models and utilities
- **TorchAudio 2.10.0** - Audio processing (if needed)

### Model & Training Utilities
- **timm 1.0.24** - Vision transformer backbones and pre-trained models
- **einops 0.8.2** - Tensor operations for transformers
- **tensorboard 2.20.0** - Training visualization and logging
- **scikit-learn 1.8.0** - Metrics, data splitting, preprocessing

### Data Processing (Already Installed)
- **MediaPipe 0.10.32** - Pose and hand landmark extraction
- **OpenCV 4.13.0** - Video processing
- **NumPy 2.4.2** - Numerical computing
- **Streamlit 1.54.0** - Web app interface

### Visualization
- **Matplotlib 3.10.8** - Plotting and visualization
- **Seaborn 0.13.2** - Statistical visualizations

---

## ✨ MPS (Metal) Acceleration Status

```
PyTorch version: 2.10.0
MPS available: ✅ True
MPS built: ✅ True
```

Your M4 MacBook's GPU is **fully configured** for PyTorch training!

---

## 🚀 Quick Start Commands

### Run the Streamlit App
```bash
source .venv/bin/activate
streamlit run keyframe_extractor/app.py
```

### Test PyTorch + MPS
```bash
/Users/nayburrahman/Documents/sem-5-ml-project/.venv/bin/python -c "
import torch
device = torch.device('mps')
x = torch.randn(3, 3, device=device)
print(f'Tensor on MPS: {x.device}')
print(f'GPU acceleration working: {x.is_mps}')
"
```

### Test RAFT (Optical Flow)
```bash
/Users/nayburrahman/Documents/sem-5-ml-project/.venv/bin/python -c "
from torchvision.models.optical_flow import raft_large
model = raft_large(pretrained=True)
print('RAFT model loaded successfully!')
print(f'Model type: {type(model).__name__}')
"
```

---

## 📁 Project Structure

```
sem-5-ml-project/
├── keyframe_extractor/        # ✅ Keyframe extraction pipeline
│   ├── app.py                 # Streamlit interface
│   ├── algorithms.py          # Extraction algorithms
│   ├── video_utils.py         # Video I/O
│   ├── landmark_extractor.py  # MediaPipe landmarks
│   └── views/                 # UI components
├── model/                     # 🔄 Model training (to be built)
│   └── WLASL_recognition_using_Action_Detection.ipynb
├── doc/                       # Documentation
├── main.py                    # Main entry point
└── pyproject.toml            # ✅ Updated with all dependencies
```

---

## 🎯 Next Steps for RAFT Integration

### Option 1: Use Pre-trained RAFT from TorchVision
```python
from torchvision.models.optical_flow import raft_large, raft_small
import torch

# Load model
model = raft_large(pretrained=True)
model = model.to('mps')  # Move to M4 GPU
model.eval()

# Extract optical flow from 2 consecutive frames
# flow = model(frame1, frame2)
```

### Option 2: Clone Official RAFT Repository
```bash
cd model/
git clone https://github.com/princeton-vl/RAFT.git
```

---

## 💡 Recommended Training Pipeline

### 1. Data Preparation
- ✅ **Already Done**: Extract keyframes from videos
- ✅ **Already Done**: Extract MediaPipe landmarks (258 features)
- 🔄 **New**: Extract RAFT optical flow between keyframe pairs

### 2. Feature Combination
Combine two feature streams:
- **Spatial**: MediaPipe landmarks (pose + hand keypoints)
- **Temporal**: RAFT optical flow (motion patterns)

### 3. Model Architecture Options

#### A) LSTM Classifier
```python
import torch.nn as nn

class SignLanguageClassifier(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.landmark_encoder = nn.LSTM(258, 128, 2, batch_first=True)
        self.flow_encoder = nn.Conv2d(2, 64, 3, padding=1)
        self.classifier = nn.Linear(128 + 64, num_classes)
    
    def forward(self, landmarks, flow):
        # Process landmarks with LSTM
        # Process flow with CNN
        # Combine features
        # Classify
```

#### B) Transformer Classifier
```python
from torch import nn
from einops.layers.torch import Rearrange

class TransformerSignClassifier(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8),
            num_layers=4
        )
        self.classifier = nn.Linear(256, num_classes)
```

---

## 🔧 M4 MacBook Optimization Tips

### 1. Enable MPS Device
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
```

### 2. Mixed Precision Training (Faster)
```python
from torch.amp import autocast, GradScaler

scaler = GradScaler('mps')

for data, target in train_loader:
    with autocast('mps'):
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 3. Optimal Batch Size
- Start with batch size 16-32
- Monitor memory usage: Activity Monitor → GPU History
- M4 has unified memory - balance between RAM and GPU

### 4. DataLoader Configuration
```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,      # M4 has efficient cores
    pin_memory=False,   # Not needed for MPS
    shuffle=True
)
```

---

## 📊 Current Dataset Status

From your notebook analysis:
- **Dataset**: WLASL15 (15 ASL sign classes)
- **Format**: 30 frames × 258 features per sample
- **Features**: 
  - Pose landmarks: 132 features (33 points × 4: x, y, z, visibility)
  - Left hand: 63 features (21 points × 3: x, y, z)
  - Right hand: 63 features (21 points × 3: x, y, z)

---

## 🐛 Troubleshooting

### If MPS is not available
```bash
# Check macOS version (needs 13+)
sw_vers

# Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

### If CUDA errors appear
```python
# MPS doesn't support all PyTorch operations yet
# Fallback to CPU for unsupported ops:
import torch
torch.set_default_device('cpu')
```

### Memory Issues
```python
# Clear MPS cache
import torch
torch.mps.empty_cache()
```

---

## 📚 Resources

- **PyTorch MPS Docs**: https://pytorch.org/docs/stable/notes/mps.html
- **RAFT Paper**: https://arxiv.org/abs/2003.12039
- **TorchVision RAFT**: https://pytorch.org/vision/main/models/raft.html
- **timm Documentation**: https://huggingface.co/docs/timm
- **MediaPipe**: https://developers.google.com/mediapipe

---

## ✅ Installation Verification

Run this complete verification:

```bash
/Users/nayburrahman/Documents/sem-5-ml-project/.venv/bin/python << 'EOF'
import sys
print("=" * 60)
print("INSTALLATION VERIFICATION")
print("=" * 60)

# 1. Check Python version
print(f"\n✓ Python: {sys.version.split()[0]}")

# 2. Check PyTorch
import torch
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ MPS Available: {torch.backends.mps.is_available()}")
print(f"✓ MPS Built: {torch.backends.mps.is_built()}")

# 3. Check TorchVision
import torchvision
print(f"✓ TorchVision: {torchvision.__version__}")

# 4. Check other packages
import timm, einops, tqdm, tensorboard, sklearn
print(f"✓ timm: {timm.__version__}")
print(f"✓ einops: {einops.__version__}")
print(f"✓ scikit-learn: {sklearn.__version__}")

# 5. Check existing packages
import mediapipe, cv2, numpy, streamlit
print(f"✓ MediaPipe: {mediapipe.__version__}")
print(f"✓ OpenCV: {cv2.__version__}")
print(f"✓ NumPy: {numpy.__version__}")
print(f"✓ Streamlit: {streamlit.__version__}")

# 6. Test MPS
print("\n" + "=" * 60)
print("MPS GPU TEST")
print("=" * 60)
device = torch.device('mps')
x = torch.randn(100, 100, device=device)
y = torch.randn(100, 100, device=device)
z = torch.matmul(x, y)
print(f"✓ Matrix multiplication on MPS: {z.shape}")
print(f"✓ Result device: {z.device}")

print("\n" + "=" * 60)
print("🎉 ALL CHECKS PASSED!")
print("=" * 60)
EOF
```

---

**Status**: ✅ Ready for model training on M4 MacBook with MPS acceleration!

**Date Completed**: February 15, 2026
