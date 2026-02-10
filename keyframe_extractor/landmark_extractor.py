"""
MediaPipe-based landmark extraction for the WLASL Keyframe & Feature Extractor.

Uses the MediaPipe Tasks API (0.10.14+) with PoseLandmarker and HandLandmarker.

Extracts Pose (33 × 4 = 132) + Left Hand (21 × 3 = 63) + Right Hand (21 × 3 = 63)
= 258 features per frame, matching the model notebook's expected shape (N, 258).
"""

import os
import urllib.request

import numpy as np
import mediapipe as mp

# --- Constants ---
NUM_POSE_LANDMARKS = 33
POSE_VALUES_PER_LANDMARK = 4  # x, y, z, visibility
POSE_FEATURE_COUNT = NUM_POSE_LANDMARKS * POSE_VALUES_PER_LANDMARK  # 132

NUM_HAND_LANDMARKS = 21
HAND_VALUES_PER_LANDMARK = 3  # x, y, z
HAND_FEATURE_COUNT = NUM_HAND_LANDMARKS * HAND_VALUES_PER_LANDMARK  # 63

TOTAL_FEATURES = POSE_FEATURE_COUNT + 2 * HAND_FEATURE_COUNT  # 258

# Landmark extraction method options
LANDMARK_METHODS = [
    "None",
    "MediaPipe Pose",
    "MediaPipe Hands",
    "MediaPipe Pose + Hands",
]

# Model URLs
_POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
_HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# Model cache directory
_MODEL_DIR = os.path.join(os.path.dirname(__file__), ".models")


def _ensure_model(url: str, filename: str) -> str:
    """Download a model file if it doesn't exist locally. Returns the local path."""
    os.makedirs(_MODEL_DIR, exist_ok=True)
    local_path = os.path.join(_MODEL_DIR, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, local_path)
        print(f"Downloaded {filename} to {local_path}")
    return local_path


def _get_pose_model_path() -> str:
    return _ensure_model(_POSE_MODEL_URL, "pose_landmarker_lite.task")


def _get_hand_model_path() -> str:
    return _ensure_model(_HAND_MODEL_URL, "hand_landmarker.task")


def _normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalizes landmark values to [0, 1] range per feature dimension.
    This removes body-size variance across different subjects.
    """
    min_vals = landmarks.min(axis=0)
    max_vals = landmarks.max(axis=0)
    range_vals = max_vals - min_vals
    # Avoid division by zero for constant features
    range_vals[range_vals == 0] = 1.0
    normalized = (landmarks - min_vals) / range_vals
    return normalized


def _localize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Translates all landmarks so the body center (average of left hip and right hip)
    is at the origin (0, 0, 0). This makes landmark data translation-invariant.

    Uses Pose landmarks 23 (left hip) and 24 (right hip).
    Each pose landmark has 4 values (x, y, z, visibility), so:
      - Left hip x starts at index 23*4 = 92
      - Right hip x starts at index 24*4 = 96
    """
    localized = landmarks.copy()

    for i in range(localized.shape[0]):
        frame = localized[i]

        # Extract hip positions (x, y, z only, skip visibility)
        left_hip_idx = 23 * POSE_VALUES_PER_LANDMARK  # 92
        right_hip_idx = 24 * POSE_VALUES_PER_LANDMARK  # 96

        left_hip = frame[left_hip_idx: left_hip_idx + 3]
        right_hip = frame[right_hip_idx: right_hip_idx + 3]

        # If both hips are zero (no pose detected), skip
        if np.all(left_hip == 0) and np.all(right_hip == 0):
            continue

        center = (left_hip + right_hip) / 2.0

        # Shift pose landmarks (x, y, z for each of 33 landmarks; skip visibility)
        for j in range(NUM_POSE_LANDMARKS):
            base = j * POSE_VALUES_PER_LANDMARK
            frame[base: base + 3] -= center

        # Shift left hand landmarks
        offset = POSE_FEATURE_COUNT
        for j in range(NUM_HAND_LANDMARKS):
            base = offset + j * HAND_VALUES_PER_LANDMARK
            frame[base: base + 3] -= center

        # Shift right hand landmarks
        offset = POSE_FEATURE_COUNT + HAND_FEATURE_COUNT
        for j in range(NUM_HAND_LANDMARKS):
            base = offset + j * HAND_VALUES_PER_LANDMARK
            frame[base: base + 3] -= center

        localized[i] = frame

    return localized


def _extract_pose_from_result(result) -> np.ndarray:
    """Extract pose landmarks from a PoseLandmarkerResult."""
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        landmarks = result.pose_landmarks[0]  # First person
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]
        ).flatten()
    else:
        pose = np.zeros(POSE_FEATURE_COUNT)
    return pose


def _extract_hands_from_result(result) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract left and right hand landmarks from a HandLandmarkerResult.
    Returns (left_hand, right_hand) as flat arrays.
    """
    left_hand = np.zeros(HAND_FEATURE_COUNT)
    right_hand = np.zeros(HAND_FEATURE_COUNT)

    if result.hand_landmarks and result.handedness:
        for i, (hand_lm, handedness_list) in enumerate(
            zip(result.hand_landmarks, result.handedness)
        ):
            # handedness_list is a list of Category; take the first one
            label = handedness_list[0].category_name
            flat = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm]).flatten()
            if label == "Left":
                left_hand = flat
            elif label == "Right":
                right_hand = flat

    return left_hand, right_hand


def extract_landmarks_from_frames(
    frames: list,
    method: str = "MediaPipe Pose + Hands",
    normalize: bool = False,
    localize: bool = False,
) -> np.ndarray:
    """
    Extracts landmarks from a list of RGB image frames using MediaPipe Tasks API.

    Args:
        frames: List of RGB image frames (numpy arrays).
        method: One of the LANDMARK_METHODS options.
        normalize: If True, normalize landmarks to [0, 1] range.
        localize: If True, translate landmarks so body center is at origin.

    Returns:
        np.ndarray of shape (num_frames, 258) containing extracted landmarks.
    """
    if method == "None":
        return np.zeros((len(frames), TOTAL_FEATURES))

    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Set up landmarkers based on method
    pose_landmarker = None
    hand_landmarker = None

    use_pose = method in ("MediaPipe Pose", "MediaPipe Pose + Hands")
    use_hands = method in ("MediaPipe Hands", "MediaPipe Pose + Hands")

    if use_pose:
        pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_get_pose_model_path()),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
        )
        pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)

    if use_hands:
        hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_get_hand_model_path()),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
        )
        hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)

    all_landmarks = []

    try:
        for frame in frames:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            pose = np.zeros(POSE_FEATURE_COUNT)
            left_hand = np.zeros(HAND_FEATURE_COUNT)
            right_hand = np.zeros(HAND_FEATURE_COUNT)

            if pose_landmarker:
                pose_result = pose_landmarker.detect(mp_image)
                pose = _extract_pose_from_result(pose_result)

            if hand_landmarker:
                hand_result = hand_landmarker.detect(mp_image)
                left_hand, right_hand = _extract_hands_from_result(hand_result)

            frame_landmarks = np.concatenate([pose, left_hand, right_hand])
            all_landmarks.append(frame_landmarks)
    finally:
        if pose_landmarker:
            pose_landmarker.close()
        if hand_landmarker:
            hand_landmarker.close()

    landmarks_array = np.array(all_landmarks, dtype=np.float32)

    # Apply optional transformations
    if localize and method in ("MediaPipe Pose", "MediaPipe Pose + Hands"):
        landmarks_array = _localize_landmarks(landmarks_array)

    if normalize:
        landmarks_array = _normalize_landmarks(landmarks_array)

    return landmarks_array
