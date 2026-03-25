# Pose2Word

### Word-Level Sign Language Recognition Workbench

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks-0F9D58)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Arrays-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![UV](https://img.shields.io/badge/UV-Package_Manager-6E56CF)](https://docs.astral.sh/uv/)

Pose2Word is an end-to-end project for preparing sign-language video data, extracting compact motion-aware features, training classifiers, and running upload-and-predict inference in one workflow.

---

## 👨‍🎓 Student Name(s)

- Navid
- Musaddiq
- Mahbub
- Raiyan
- Sinha

## 📚 Course Name

- CSE-4544: ML Lab Final Project (IUT-SWE-22)



## 🎯 Problem Statement

### What problem are you trying to solve?

Build a practical word-level sign-language recognition system that can process raw videos and predict sign classes reliably.

### Why is this problem important?

Sign-language recognition supports accessibility and improves communication support for Deaf and Hard-of-Hearing communities in digital systems.

### Real-world application of the problem

- assistive communication tools
- educational sign-learning applications
- HCI interfaces and smart interpretation systems
- rapid prototyping for accessibility-focused products

---

## 🗂️ Dataset

### Source of the dataset

- WLASL-based dataset organization
- local raw dataset root: `dataset/raw_video_data`

### Number of samples and features

- 15 word classes in the core setup
- 187 total `.mp4` video samples (current snapshot)
- extracted feature format (current primary path): `(T, 168)` where default `T = 15`

### Key attributes/features

- signer-centered cropped and normalized videos
- semantically selected keyframes (8 to 15)
- MediaPipe-based landmarks with Relative Quantization (RQ)

### Any challenges in the data

- variable signer motion and framing
- idle segments before/after signing
- class-order and sequence-length mismatch risks during inference

---

## 🧹 Data Preprocessing

### Handling missing values

- no tabular missing-value imputation is used
- for vision data, missing detections are handled by robust defaults and sequence padding/truncation

### Data cleaning

- frame-rate normalization via ffmpeg
- CLAHE-based contrast enhancement
- signer-focused cropping
- temporal idle trimming
- resize + pad to square output

### Feature engineering or selection

- keyframe extraction using fused motion/landmark signals
- landmark selection to 56 points per frame (168 dims after flattening)
- hand dominance correction and shoulder-based calibration
- relative quantization + feature scaling

### Data splitting (training/testing)

- model training scripts perform train/validation splitting during training workflow
- exact split configuration depends on selected trainer script and run config

---

## 🤖 Machine Learning Model(s)

### Algorithm(s) used

- LSTM classifier
- Transformer classifier
- Hybrid architecture support

### Why you chose these models

- sequential temporal modeling fits sign progression over frames
- LSTM provides a lightweight baseline
- Transformer/hybrid variants support richer temporal feature learning

### Brief explanation of how they work

- input is a fixed-length landmark sequence
- sequence encoder learns temporal representation
- final classifier head outputs class probabilities across sign words

---

## 🏋️ Model Training

### Training process

1. preprocess raw videos
2. extract keyframes
3. generate landmark tensors
4. train classifier and save checkpoints/logs

### Important parameters (hyperparameters)

- `model_type`: `lstm` / `transformer` / `hybrid`
- `batch_size` (example default in pipeline runner: 8)
- `num_epochs` (example default: 50)
- `learning_rate` (example default: 0.001)
- `weight_decay` (example default: 1e-4)
- `target_seq_len` (landmark default often 15)
- device: `cuda` / `mps` / `cpu`

### Tools/libraries used

- Python
- PyTorch ecosystem
- Streamlit (for app interactions)
- MediaPipe Tasks
- OpenCV
- NumPy / SciPy / scikit-learn

---

## 📈 Results and Evaluation

### Evaluation metrics

- accuracy (tracked in available training history)
- training and validation loss curves

### Available run snapshot (example from current checkpoints)

From `mini_brother_mother_quick5/training_history.json`:

- `val_acc` reached `1.0` by epoch 4 to 5
- `val_loss` improved from ~`0.6933` to ~`0.6895`

### Comparison if multiple models were used

- multiple model types are supported by the training API
- side-by-side benchmark tables are not yet consolidated in the repository docs

---

## 🖼️ Visualization / Demo

- Streamlit tabs provide visual workflow demos:
  - preprocessing page
  - keyframe preview and selected frame outputs
  - landmark extraction outputs (`.npy`)
  - prediction output with top-k probabilities and confidence
- TensorBoard logs are generated in checkpoint run folders for training visualization

---

## ⚠️ Challenges and Limitations

### Problems faced during the project

- keeping training and inference sequence settings aligned
- handling variable-quality videos and inconsistent motion windows
- balancing extraction quality vs processing speed

### Limitations of the current solution

- closed-vocabulary prediction (only trained classes)
- no full benchmark report table for all runs in one place
- no dedicated in-app training tab (training is CLI-driven)

---

## ✅ Conclusion and Future Work

### Key findings

- a unified preprocessing + landmark + training + inference pipeline improves reproducibility
- landmark-first representation works as a strong practical baseline
- end-to-end tooling in one repo reduces integration errors

### Possible improvements

- standardized experiment tracking and comparison dashboard
- expanded dataset scale and class coverage
- stronger model selection and calibration workflows

### Future research directions

- robust signer/domain adaptation
- improved temporal attention variants
- multimodal feature fusion extensions

---

## 🧠 We Kept Words (Dataset Snapshot)

Based on WLASL, we kept 15 word classes:

| Word (Class) | Videos (.mp4) |
|---|---:|
| brother | 11 |
| call | 12 |
| drink | 15 |
| go | 15 |
| help | 14 |
| man | 12 |
| mother | 11 |
| no | 11 |
| short | 13 |
| tall | 13 |
| what | 12 |
| who | 14 |
| why | 11 |
| woman | 11 |
| yes | 12 |
| **TOTAL MP4s** | **187** |

---

## 🚀 Quick Start

```bash
uv sync
uv run streamlit run main.py
```

### End-to-end CLI pipeline

```bash
uv run python util_scripts/run_full_pipeline.py \
  --dataset-dir dataset/raw_video_data \
  --preprocessed-dir outputs/preprocessed \
  --keyframes-dir outputs/keyframes \
  --landmarks-dir outputs/landmarks \
  --checkpoint-dir checkpoints/full_pipeline_run \
  --trainer-script model/trainer_current_config-gpu.py \
  --model-type lstm \
  --device cuda
```

---

## 📌 Notes

- Prediction is closed-vocabulary.
- Class label order must match training order exactly.
- Sequence length in prediction must match training configuration.
- If ffmpeg is unavailable, normalization and prediction preprocessing will fail.
