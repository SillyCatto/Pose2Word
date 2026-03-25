# Pose2Word Codebase Report

This report is based on the source code in the repository, starting from `main.py` and tracing how execution flows through the project. Existing markdown documentation was intentionally ignored while producing this report.

## 1. What This Project Actually Is

At a practical level, this repository is a sign-language data preparation and recognition workbench centered on word-level ASL/WLASL-style videos.

Its current codebase does four main jobs:

1. Preprocess raw sign-language videos into standardized square `.mp4` clips.
2. Extract a small set of semantically important keyframes from those videos.
3. Convert keyframes into fixed-size landmark feature tensors suitable for training.
4. Load a trained sequence classifier and predict a sign from an uploaded video.

The project is not just a model or just a UI. It is a pipeline toolchain:

- `app/` contains the production-facing Streamlit app and the actual preprocessing / extraction / inference logic.
- `model/` contains dataset loaders, model definitions, trainers, and one experimental optical-flow utility.
- `util_scripts/` contains orchestration and demo scripts that chain the pipeline together outside the UI.

The codebase is currently most coherent as a landmark-first pipeline. Although several files still mention RAFT optical flow, the active training and inference flow is mostly based on MediaPipe landmarks, not RAFT features.

## 2. High-Level Architecture

### Core end-to-end training data path

```text
Raw videos
  -> preprocessing pipeline
  -> preprocessed square videos
  -> keyframe extraction
  -> per-video keyframe image folders
  -> landmark extraction
  -> per-video .npy feature files
  -> training scripts
  -> checkpoint (.pth)
```

### Core prediction path

```text
Uploaded video
  -> preprocessing subset
  -> feature extraction
  -> trained checkpoint
  -> top-k class probabilities
```

### Main packages and their responsibilities

- `app.views`: Streamlit tabs and user interaction.
- `app.preprocessing`: Raw video normalization, lighting normalization, signer crop, trimming, output writing, metadata, quality checks.
- `app.core.keyframe_pipeline`: Hybrid signal-based keyframe extraction.
- `app.core.landmark_pipeline`: Landmark selection, dominant-hand normalization, relative quantization, fixed-length sequence generation.
- `app.core.inference`: Upload-to-prediction path using trained checkpoints.
- `model`: Offline model creation, dataset loading, and training.

## 3. Execution Flow From `main.py`

`main.py` is the Streamlit entrypoint.

When the app starts:

1. It imports the four page modules from `app.views`.
2. It sets Streamlit page configuration and some custom CSS for tab styling.
3. It creates four tabs:
   - Video Preprocessing
   - Keyframe Extractor
   - Landmark Extractor
   - Predict Sign
4. It calls `render()` on each page module inside its corresponding tab.

Important practical detail: this is a Streamlit app, so the script reruns frequently. The page modules therefore store results and form values in `st.session_state` and only perform heavy work when the user clicks a button.

### Tab 1: Preprocessing

`main.py -> app.views.preprocessing_page.render()`

This tab is a UI wrapper around `app.preprocessing.PreprocessingPipeline`.

It supports:

- Single-video processing
- Batch dataset processing
- Quality report generation

Single-video mode can either:

- upload a file, or
- pick a dataset sample from `dataset/raw_video_data/labels.csv`

The actual work happens in `PreprocessingPipeline.process_single_video()`, which runs a 9-step preprocessing pipeline and produces:

- a preprocessed `.mp4`
- metadata about the crop, trim, frame counts, and timestamp

Batch mode loops over `labels.csv` rows and calls the same single-video pipeline for each sample.

Quality report mode scans already-processed outputs and checks whether the generated videos exist, can be opened, and have enough frames.

### Tab 2: Keyframe Extractor

`main.py -> app.views.keyframe_page.render()`

This tab is a UI wrapper around `app.core.keyframe_pipeline`.

It supports:

- Single uploaded video -> extract keyframes -> preview -> save images
- Batch directory processing

The core call is `app.core.keyframe_pipeline.run_pipeline(video_path, config)`.

The output is a `PipelineResult` containing:

- selected frame indices
- timestamps
- keyframe images
- fused motion/hold signals
- simple detection stats

Batch mode expects a dataset layout of:

```text
root/
  label/
    video.mp4
```

and writes:

```text
outputs/keyframes/
  label/
    video_stem/
      frame_0.png
      frame_1.png
      ...
```

### Tab 3: Landmark Extractor

`main.py -> app.views.landmark_page.render()`

This tab wraps `app.core.landmark_pipeline`.

It supports:

- Single keyframe directory -> landmark extraction -> preview -> save `.npy`
- Batch processing of the full extracted-keyframe dataset

The core call is `app.core.landmark_pipeline.run_pipeline(frames_dir, config)`.

The output is a `LandmarkResult` containing:

- the processed landmark tensor
- original frame count
- dominant hand
- detection counts

Batch mode expects:

```text
outputs/keyframes/
  label/
    video_stem/
      frame_0.png
      ...
```

and writes:

```text
outputs/landmarks/
  label/
    video_stem.npy
```

### Tab 4: Predict Sign

`main.py -> app.views.predict_page.render()`

This tab is the inference surface.

It:

1. Accepts an uploaded video.
2. Accepts a checkpoint path.
3. Accepts class labels and a confidence threshold.
4. Calls `app.core.inference.predict_video(...)`.

That function:

1. Loads the checkpoint and rebuilds the matching model.
2. Preprocesses the uploaded video.
3. Extracts model features.
4. Runs the model.
5. Returns ranked predictions and preprocessing metadata.

This predictor is explicitly closed-vocabulary. It can only predict among the classes represented by the loaded checkpoint.

## 4. What the Preprocessing Pipeline Actually Does

The preprocessing subsystem lives under `app/preprocessing/` and is orchestrated by `app/preprocessing/runner.py`.

### The real 9-step pipeline

`PreprocessingPipeline._process()` labels its steps like this:

1. Probe source FPS with OpenCV.
2. Normalize the video with `ffmpeg` and extract frames from the normalized file.
3. Apply CLAHE to each frame for lighting normalization.
4. Run MediaPipe Pose over the frames to estimate a signer crop and collect wrist-based trim signals.
5. Crop every frame to the signer box.
6. Resize with letterboxing to a square output size.
7. Trim inactive head/tail regions.
8. Write the output video.
9. Update global metadata.

### Step-by-step details

#### Step 1: Video normalization

`app/preprocessing/video_normalizer.py`

The pipeline first forces videos into a consistent format:

- constant FPS
- H.264 video
- `yuv420p`
- no audio

This matters because later motion-sensitive stages assume stable frame timing.

The implementation prefers a system `ffmpeg`, then falls back to `imageio-ffmpeg` if necessary.

#### Step 2: Frame extraction inside the normalization stage

`app/preprocessing/frame_extraction.py`

After normalization, the temporary normalized video is decoded with OpenCV into a list of RGB `numpy` arrays. The code keeps frames in memory for the rest of the pipeline.

#### Step 3: CLAHE

`app/preprocessing/clahe.py`

CLAHE is applied to the luminance channel in LAB space. This is an image normalization step designed to make MediaPipe more stable across uneven lighting conditions.

#### Step 4: Signer localization

`app/preprocessing/signer_crop.py`

This is one of the most important modules in the repo.

It runs MediaPipe Pose over every frame and extracts two outputs in one pass:

- an adaptive crop bounding box around the signer
- wrist-based signals for later temporal trimming

The crop logic is more sophisticated than a simple median body box. It uses:

- visible pose anchors
- a forehead proxy derived from nose/eye landmarks
- wrist locations
- estimated fingertip reach by extending elbow-to-wrist direction
- asymmetric side/top/bottom margins
- a minimum crop-size safeguard

This design is meant to reduce clipping when hands rise above the head.

#### Step 5: Crop

`crop_frames()` slices every frame with the computed bounding box.

#### Step 6: Resize and pad

`app/preprocessing/video_writer.py -> resize_with_padding()`

Frames are resized so the longer side fits `output_size`, then black-padded to produce a square frame without distortion.

#### Step 7: Temporal trim

`app/preprocessing/temporal_trim.py`

The code computes per-frame wrist velocity and uses two cues to find active signing:

- whether a wrist is visible
- whether wrist motion is above a threshold

The goal is to remove idle frames before and after the sign.

Actual implementation behavior:

- It computes an `idle` boolean as `no hand visible OR low wrist velocity`.
- It scans forward and backward looking for runs with at least 3 active frames.
- It adds a 2-frame safety buffer on both ends.
- If the remaining clip would be too short, it keeps the whole sequence.

Important implementation note: the module comments describe a duration-gated idle detector, and `min_idle_frames` is computed, but that value is not actually used in the trimming logic. In other words, the code currently trims by active-run detection plus a minimum-length safeguard, not by the documented sustained-idle-duration rule.

#### Step 8: Video writing

`app/preprocessing/video_writer.py -> write_video()`

The processed RGB frames are streamed directly into `ffmpeg` through stdin and encoded as H.264 `.mp4`. This avoids writing image-frame intermediates.

#### Step 9: Metadata and quality checks

`app/preprocessing/storage.py` and `app/preprocessing/quality_checks.py`

Metadata is written to a shared `metadata.json` under the preprocessing output root. Each sample entry stores:

- source video name
- label
- raw and output FPS
- original and trimmed frame counts
- trim range
- crop box and crop size
- output size
- pipeline version
- processing timestamp

Quality checks later verify:

- the output video exists
- OpenCV can open it
- it contains at least the minimum number of frames

### Preprocessing output contract

Current preprocessing output is:

```text
outputs/preprocessed/
  metadata.json
  label/
    video_stem.mp4
```

This is important because some older notebooks in the repo assume a richer per-sample artifact folder that the current preprocessing code does not generate.

## 5. What the Keyframe Pipeline Actually Does

The keyframe subsystem lives in `app/core/keyframe_pipeline.py`.

Its goal is not to uniformly sample frames. It tries to identify a small number of meaningful frames from a sign sequence.

### Inputs and outputs

Input:

- one video file

Output:

- 8 to 15 selected keyframes by default
- per-frame fused scoring signals
- keyframe images
- metadata such as FPS and hand-detection count

### The four signals

The keyframe selector fuses four signals:

1. Farneback optical-flow magnitude
2. Wrist velocity from MediaPipe hand landmarks
3. Handshape change from MediaPipe hand landmarks
4. Flow-direction reversal

### How it works internally

#### Video loading

`load_video()` reads the full video with OpenCV into memory as BGR frames.

#### MediaPipe hand landmarks

`LandmarkSignalExtractor`

This uses `HandVideoLandmarker` from `app/core/mediapipe_tasks.py`, which runs MediaPipe Tasks in VIDEO mode so temporal tracking can be reused across frames.

From each frame it derives:

- left-hand landmarks
- right-hand landmarks
- merged hand bounding box
- a 126-value landmark vector for both hands combined

From those landmarks it then computes:

- wrist velocity across frames
- handshape-change score based on non-wrist hand landmarks

#### Optical flow

`FlowSignalExtractor`

Dense Farneback optical flow is computed between consecutive frames. If a hand bounding box is available, flow statistics are restricted to the hand region plus some padding.

It extracts:

- average motion magnitude
- average motion direction change

#### Signal fusion

`fuse_signals()`

Each signal is normalized and smoothed. Then:

- `fused_transition` is a weighted combination of flow magnitude, flow direction, wrist velocity, and handshape change
- `fused_hold` is an inverse stillness-style signal derived from tightly smoothed wrist and handshape changes

#### Keyframe selection

`select_keyframes()`

The algorithm combines several heuristics:

- always include first and last frame
- take peaks of transition signal
- take peaks of hold signal
- detect “sandwiched holds” by finding deep valleys between motion peaks
- if too many frames are selected, drop low-scoring middle frames
- if too few are selected, add high-scoring unselected frames

This makes the extractor deliberately favor:

- movement transitions
- held poses
- short meaningful pauses between motion bursts

### Keyframe output contract

Current keyframe output is:

```text
outputs/keyframes/
  label/
    video_stem/
      frame_0.png
      frame_1.png
      ...
```

The saved filenames are rank-based (`frame_0`, `frame_1`, ...), not original frame-number-based.

## 6. What the Landmark Pipeline Actually Does

The landmark subsystem lives in `app/core/landmark_pipeline.py`.

Its job is to turn a variable-length folder of extracted keyframe images into a fixed-size numeric representation suitable for training.

### Landmark representation

The current compact feature vector is 168 values per frame:

- left hand: 21 landmarks x 3 = 63
- right hand: 21 landmarks x 3 = 63
- pose: 8 selected landmarks x 3 = 24
- face: 6 selected anchor landmarks x 3 = 18

Total: `63 + 63 + 24 + 18 = 168`

### The processing stages

#### Stage 1: Load keyframe images

`load_keyframe_images()`

Loads `frame_*.png` / `frame_*.jpg` and sorts them by the number embedded in the filename.

#### Stage 2: Landmark extraction

`extract_landmarks()`

This uses `HolisticLikeImageLandmarker` from `app/core/mediapipe_tasks.py`.

That class is a composed wrapper around three MediaPipe Tasks detectors:

- PoseLandmarker
- HandLandmarker
- FaceLandmarker

Each image is processed independently in IMAGE mode.

Only selected subsets are retained:

- all 21 landmarks from each hand
- 8 pose landmarks
- 6 face landmarks

#### Stage 3: Hand dominance correction

`correct_hand_dominance()`

The code detects which hand is more active over the sequence. If the signer appears left-dominant, the sequence is mirrored and left/right slots are swapped so the dominant hand is always represented in the right-hand portion of the feature vector.

This normalization also swaps:

- left/right pose pairs
- left/right face-cheek anchors

#### Stage 4: Shoulder-based calibration

`calibrate_to_shoulders()`

All coordinates are translated so the midpoint of the first frame’s shoulders becomes the origin. This makes coordinates more body-relative and less camera-relative.

#### Stage 5: Relative quantization

`relative_quantize()`

This is the pipeline’s most research-specific transformation.

Different landmark groups are translated to local centers and then quantized separately:

- left hand relative to left wrist
- right hand relative to right wrist
- pose limbs relative to shoulder midpoint
- face relative to nose tip

Shoulders themselves are kept as calibrated global coordinates.

Per-axis quantization levels differ by region:

- hands: `(10, 10, 5)`
- face: `(5, 5, 3)`
- pose: `(10, 10, 5)`

#### Stage 6: Scaling and fixed-length shaping

After quantization:

- features are multiplied by a constant scale factor, default `100`
- the sequence is padded or truncated to a fixed target length

The default UI target length is 15.

### Landmark output contract

Current landmark output is:

```text
outputs/landmarks/
  label/
    video_stem.npy
```

Each file stores a tensor with shape:

```text
(target_sequence_length, 168)
```

## 7. How Prediction Actually Works

Prediction logic lives in `app/core/inference.py`.

This is an important module because it connects the preprocessing stack to the model stack.

### Checkpoint loading

`load_checkpoint()`:

- loads a saved `.pth`
- reads checkpoint config
- reconstructs the model using `model.sign_classifier.create_model()`
- loads weights
- resolves class names

Supported model families:

- `lstm`
- `transformer`
- `hybrid`

### Feature extraction for inference

`build_inference_features()`

This reuses the preprocessing path:

1. normalize video
2. extract frames
3. apply CLAHE
4. analyze signer
5. crop
6. resize/pad
7. trim

Then it branches into one of two feature modes:

#### Mode A: Current RQ-landmark path (`feature_dim == 168`)

It runs:

- `extract_landmarks()`
- `correct_hand_dominance()`
- `calibrate_to_shoulders()`
- `relative_quantize()`
- `scale_features()`
- `pad_sequence()`

#### Mode B: Legacy pose+hands path (`feature_dim == 258`)

It extracts:

- pose landmarks with visibility
- left-hand landmarks
- right-hand landmarks

This produces the older `258`-dimensional representation:

- pose: `33 x 4 = 132`
- left hand: `21 x 3 = 63`
- right hand: `21 x 3 = 63`

### Prediction output

The predictor returns:

- best class label
- class index
- confidence
- top-k ranked predictions
- a confidence/pass threshold flag
- preprocessing metadata

### Important architectural reality

The current prediction path does **not** run the keyframe extractor before landmark extraction. It extracts features from the trimmed frame sequence directly.

That means the default prediction path is not identical to the training-data path implied by:

```text
preprocessed video -> keyframe extraction -> landmark extraction
```

This is one of the most important architectural mismatches in the repository.

### Another important reality

The prediction UI defaults to a target sequence length of `32`, while the landmark-extraction UI defaults to `15`. If a checkpoint was trained on 15-frame keyframe-based tensors but the predictor is run with 32-frame direct-frame features, behavior may not match training-time assumptions.

## 8. MediaPipe Task Model Management

MediaPipe integration is abstracted mainly through `app/core/mediapipe_tasks.py`.

### What this module does

It:

- ensures `.task` model files exist under `app/models/`
- wraps MediaPipe Tasks detectors
- converts detector outputs into simple Python/Numpy-friendly structures

### Two main wrappers

#### `HandVideoLandmarker`

Used by the keyframe pipeline.

- running mode: `VIDEO`
- detector: hand-only
- purpose: temporally consistent hand tracking for motion signals

#### `HolisticLikeImageLandmarker`

Used by the landmark pipeline.

- running mode: `IMAGE`
- detectors: pose + hands + face
- purpose: preserve the old “holistic-like” contract using separate Tasks APIs

### Runtime assets

The repo already includes these task-model binaries:

- `app/models/hand_landmarker.task`
- `app/models/face_landmarker.task`
- `app/models/pose_landmarker_lite.task`
- `app/models/pose_landmarker_full.task`

If missing, some modules can also download task files automatically. Complexity level `2` for pose expects a heavy model that is not checked into the repository and would be downloaded on demand.

## 9. Training and Offline Tooling

The training side lives mostly under `model/`.

### `model/sign_classifier.py`

This defines three sequence classifiers:

- `LSTMSignClassifier`
- `TransformerSignClassifier`
- `HybridSignClassifier`

All three operate on landmark sequences. None of them currently consume RAFT flow features in their forward pass, despite several docstrings still referencing RAFT.

#### LSTM model

- sequence encoder: LSTM
- sequence summary: final hidden state
- head: small MLP classifier

#### Transformer model

- linear input projection
- sinusoidal positional encoding
- transformer encoder stack
- temporal mean pooling
- MLP classifier head

#### Hybrid model

- bidirectional LSTM encoder
- self-attention over encoded sequence
- temporal mean pooling
- classifier head

### `model/dataset.py`

This loads landmark `.npy` files from class-organized directories.

It supports two layouts:

1. legacy flat format: `class/*.npy`
2. nested compatibility format: `class/sample/keypoints.npy`

It returns:

- landmark tensor
- integer class label

`create_data_loaders()` performs a random train/validation split, default `80/20`.

There is no dedicated test split in the core training scripts by default.

### `model/trainer.py`

This is the simpler trainer, oriented toward the original project setup and MPS-friendly training.

It provides:

- mixed precision where supported
- TensorBoard logging
- periodic checkpoint saving
- training-history JSON export

### `model/trainer_current_config-gpu.py`

This is a more current trainer tuned for CUDA / RTX-class GPUs.

Compared with `trainer.py`, it adds:

- gradient accumulation
- gradient clipping
- CUDA pinning/persistent workers
- optional periodic CUDA cache clearing
- input-shape inference from the dataset
- richer checkpoint config metadata

The full pipeline runner defaults to this trainer script, which suggests it is the preferred current training script.

### `model/raft_flow_extractor.py`

This is a standalone experimental RAFT optical-flow utility.

It can:

- load pretrained `raft_small` / `raft_large`
- compute flow fields between frames
- convert flow to magnitude/angle
- pool flow features spatially
- save visualizations

However, it is not currently wired into:

- the Streamlit app
- the main landmark training path
- the main inference path

So in the current repository state, RAFT is more of an experimental side module than a core production dependency.

## 10. Root-Level and Script Entry Points

### `main.py`

The Streamlit app entrypoint. This is the main interactive surface.

### `app/preprocessing/__main__.py`

CLI entrypoint for preprocessing.

It supports:

- processing the whole dataset
- processing one label
- printing a report
- changing output size / FPS / CLAHE

Notably, this CLI batch mode scans the filesystem for `.mp4` files instead of relying on `labels.csv`.

### `util_scripts/run_full_pipeline.py`

This is the offline orchestration script for the whole workflow:

1. preprocessing
2. keyframe extraction
3. landmark extraction
4. training

It dynamically imports whichever trainer script you point it at, defaulting to `model/trainer_current_config-gpu.py`.

### `util_scripts/demo_pipeline.py`

A demo script that:

- checks the environment
- demonstrates RAFT extraction
- loads the landmark dataset
- builds models
- explains training entry points

### `util_scripts/verify_setup.py`

An environment verification script focused on:

- PyTorch
- MPS
- TorchVision
- MediaPipe
- RAFT importability

## 11. File-by-File Map

This section summarizes how the codebase fits together as a whole.

### Root files

| File | Role | How it relates |
|---|---|---|
| `main.py` | Streamlit app entrypoint | Imports all page modules and defines the main UI tabs. |
| `pyproject.toml` | Project/dependency manifest | Declares Python version, package list, and installable packages. |
| `uv.lock` | Lockfile | Pins dependency resolution for reproducible installs. |
| `CODEBASE_REPORT.md` | This report | Not part of runtime. |

### `app/`

| File | Role | How it relates |
|---|---|---|
| `app/__init__.py` | Package marker | Labels the app package. |
| `app/views/__init__.py` | View package export | Re-exports page modules used by `main.py`. |
| `app/views/preprocessing_page.py` | Preprocessing tab UI | Builds UI controls and calls `PreprocessingPipeline`. |
| `app/views/keyframe_page.py` | Keyframe tab UI | Calls `app.core.keyframe_pipeline`. |
| `app/views/landmark_page.py` | Landmark tab UI | Calls `app.core.landmark_pipeline`. |
| `app/views/predict_page.py` | Prediction tab UI | Calls `app.core.inference.predict_video`. |
| `app/views/_folder_browser.py` | Local folder picker helper | Uses `tkinter` to let the UI browse filesystem paths. |
| `app/preprocessing/__init__.py` | Public preprocessing API | Exports `PipelineConfig` and `PreprocessingPipeline`. |
| `app/preprocessing/__main__.py` | CLI preprocessing entrypoint | Batch/single-label/report command-line wrapper. |
| `app/preprocessing/config.py` | Preprocessing configuration dataclass | Central place for all preprocessing parameters. |
| `app/preprocessing/frame_extraction.py` | Video decode and FPS probe | Used by preprocessing and inference. |
| `app/preprocessing/video_normalizer.py` | ffmpeg normalization helpers | Used by preprocessing and inference. |
| `app/preprocessing/clahe.py` | Lighting normalization | Used by preprocessing and inference. |
| `app/preprocessing/signer_crop.py` | Signer localization and trim-signal extraction | One of the main algorithmic modules used by preprocessing and inference. |
| `app/preprocessing/temporal_trim.py` | Idle-boundary trimming | Used by preprocessing and inference. |
| `app/preprocessing/video_writer.py` | Resize/pad and encode output video | Used by preprocessing. |
| `app/preprocessing/storage.py` | Metadata persistence | Used by preprocessing and quality checks. |
| `app/preprocessing/quality_checks.py` | Post-run validation | Used by the preprocessing UI and CLI report mode. |
| `app/preprocessing/runner.py` | Preprocessing orchestrator | Central execution logic for the 9-step preprocessing pipeline. |
| `app/core/__init__.py` | Core package marker | Documents main business-logic modules. |
| `app/core/mediapipe_tasks.py` | MediaPipe Tasks wrappers and model management | Shared by keyframe and landmark pipelines. |
| `app/core/keyframe_pipeline.py` | Hybrid keyframe extraction algorithm | Used by the keyframe UI and full-pipeline script. |
| `app/core/landmark_pipeline.py` | Landmark extraction + normalization pipeline | Used by landmark UI, inference helpers, and full-pipeline script. |
| `app/core/inference.py` | Checkpoint loading and prediction pipeline | Used by the prediction UI. |

### `app/models/`

| File | Role | How it relates |
|---|---|---|
| `app/models/hand_landmarker.task` | MediaPipe hand model asset | Used by keyframe and landmark extraction. |
| `app/models/face_landmarker.task` | MediaPipe face model asset | Used by landmark extraction. |
| `app/models/pose_landmarker_lite.task` | MediaPipe pose model asset | Used by signer localization and some default pose tasks. |
| `app/models/pose_landmarker_full.task` | MediaPipe full pose model asset | Used when pose complexity is set to the balanced/full option. |

### `model/`

| File | Role | How it relates |
|---|---|---|
| `model/__init__.py` | Model package export | Re-exports dataset/model/trainer utilities. |
| `model/sign_classifier.py` | Sequence-model definitions | Used by training scripts and inference checkpoint loading. |
| `model/dataset.py` | Landmark dataset loader | Used by trainers and notebook tooling. |
| `model/trainer.py` | Basic trainer | Original/main generic trainer. |
| `model/trainer_current_config-gpu.py` | More current GPU trainer | Preferred by the full-pipeline runner. |
| `model/raft_flow_extractor.py` | Experimental optical-flow utility | Standalone RAFT-based feature extraction helper. |
| `model/train_transformer.ipynb` | Notebook-only transformer training experiment | Trains a transformer directly from landmark `.npy` files. |
| `model/WLASL_recognition_using_Action_Detection.ipynb` | Project exploration notebook | Contains analysis, augmentation, and training experimentation in notebook form. |

### `util_scripts/`

| File | Role | How it relates |
|---|---|---|
| `util_scripts/demo_pipeline.py` | Demonstration script | Shows environment setup, RAFT demo, dataset loading, and model creation. |
| `util_scripts/run_full_pipeline.py` | Full offline pipeline runner | Choreographs preprocessing, keyframes, landmarks, and training. |
| `util_scripts/verify_setup.py` | Environment verification | Sanity-checks PyTorch, MediaPipe, and RAFT imports. |
| `util_scripts/run_full_pipeline.ipynb` | Notebook wrapper around full pipeline | Runs the CLI pipeline and visualizes training metrics/confusion matrices. |
| `util_scripts/inspect_preprocessed.ipynb` | Inspection notebook for older artifact layout | Assumes a richer preprocessing output format than the current code produces. |

## 12. Current Data and Directory Assumptions

### Raw dataset assumptions

The UI-side preprocessing flow assumes:

```text
dataset/raw_video_data/
  labels.csv
  label/
    video.mp4
```

`labels.csv` is treated as the canonical sample list in the preprocessing UI.

### Preprocessed video assumptions

The rest of the pipeline assumes:

```text
outputs/preprocessed/
  label/
    video_stem.mp4
```

### Keyframe assumptions

```text
outputs/keyframes/
  label/
    video_stem/
      frame_0.png
      ...
```

### Landmark assumptions

```text
outputs/landmarks/
  label/
    video_stem.npy
```

### Checkpoint assumptions

Training scripts typically write:

```text
checkpoints/run_name/
  best_model.pth
  checkpoint_epoch_10.pth
  training_history.json
  logs/
```

## 13. Notable Mismatches, Legacy Pieces, and Architectural Friction

These are the most important “how it really works” caveats in the current codebase.

### 1. The repo still talks about RAFT more than it actually uses RAFT

Several model and project descriptions say “RAFT + landmarks,” but the live app, current trainers, and inference path are fundamentally landmark-based.

Evidence in code:

- `flow_dir` is explicitly deprecated/ignored in trainers.
- model forward passes consume landmark tensors only.
- inference only extracts landmark-style features, not RAFT features.
- RAFT exists as a standalone utility, not as an integrated training/inference dependency.

### 2. Training path and prediction path are not fully aligned

The most obvious intended training path is:

```text
preprocessed videos -> keyframes -> landmark pipeline -> .npy -> training
```

But the prediction path for 168-dim features is:

```text
uploaded video -> preprocessing subset -> direct landmark extraction on trimmed frames
```

It does not call the keyframe extractor.

That means a checkpoint trained on keyframe-selected landmark sequences may receive a somewhat different feature distribution at inference time.

### 3. The default sequence length in prediction may not match training

The prediction UI defaults to `32` frames. The landmark extraction UI defaults to `15`. The inference function will honor the UI-provided value, not automatically mirror the checkpoint config unless the caller leaves it unspecified.

So the user can easily run inference with a sequence length different from training.

### 4. Older checkpoints may be less inference-friendly

`trainer_current_config-gpu.py` stores richer metadata such as:

- `landmark_dim`
- `target_sequence_length`

But `trainer.py` stores a much thinner config. The inference loader relies on checkpoint config to reconstruct model input shape, so checkpoints from the older trainer may be less self-describing and therefore less reliable for automatic inference reconstruction.

### 5. Some notebook tooling targets an older artifact format

`util_scripts/inspect_preprocessed.ipynb` expects files like:

- `frames.npy`
- `keypoints.npy`
- `flow_vectors.npy`
- `masks.npy`
- `attention_mask.npy`

The current preprocessing code does not generate those artifacts. It produces a preprocessed `.mp4` plus shared metadata.

So that notebook reflects an older or alternate pipeline, not the current active preprocessing output contract.

### 6. Batch behavior is not fully unified across interfaces

- Preprocessing UI batch mode uses `labels.csv`.
- Preprocessing CLI batch mode scans for `.mp4` files recursively.
- Keyframe and landmark batch modes scan directory structure directly.

The codebase therefore has multiple batch-enumeration strategies rather than a single canonical dataset-index mechanism.

### 7. Some functionality exists but is not on the main path

Examples:

- `normalized_video_dir` exists in config but is not central to the main single-video preprocessing flow.
- `normalize_all_videos()` exists but is not what the main pipeline uses.
- MediaPipe model-download helpers are duplicated across modules.

## 14. How the Pieces Fit Together as a Whole

The cleanest way to understand this codebase is:

- `main.py` is the interactive shell.
- `app/preprocessing` standardizes raw sign videos.
- `app/core/keyframe_pipeline` compresses a video into a small set of meaningful frames.
- `app/core/landmark_pipeline` converts those frames into compact, fixed-size numerical sequences.
- `model/` trains sequence classifiers on those numerical sequences.
- `app/core/inference.py` loads those trained models and produces predictions from uploaded videos.
- `util_scripts/` automates or demonstrates the pipeline outside the UI.

So the project is really a staged representation-learning pipeline:

```text
video
  -> cleaned video
  -> salient frames
  -> landmark sequence
  -> classifier-ready tensor
  -> word label
```

## 15. Bottom-Line Summary

If you strip away naming inconsistencies and legacy artifacts, the current repository is best described as:

> A Streamlit-based sign-language dataset-preparation and closed-vocabulary recognition toolkit built around MediaPipe-based preprocessing, keyframe extraction, compact landmark sequence generation, and PyTorch sequence classifiers.

What it does best today:

- preprocess raw videos into consistent training inputs
- extract informative keyframes
- derive compact landmark tensors
- train landmark-sequence classifiers
- run simple closed-vocabulary inference from a checkpoint

What is still transitional:

- RAFT references vs actual usage
- training/inference feature-path alignment
- notebook artifacts targeting older formats
- config consistency across old vs new trainers

That is the real shape of the codebase as implemented.
