from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np


_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)
_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/"
    "face_landmarker.task"
)
_POSE_MODEL_SPECS = {
    0: (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/"
        "pose_landmarker_lite.task",
    ),
    1: (
        "pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/latest/"
        "pose_landmarker_full.task",
    ),
    2: (
        "pose_landmarker_heavy.task",
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_heavy/float16/latest/"
        "pose_landmarker_heavy.task",
    ),
}


@dataclass(frozen=True)
class HolisticLikeResult:
    left_hand_landmarks: Sequence | None
    right_hand_landmarks: Sequence | None
    pose_landmarks: Sequence | None
    face_landmarks: Sequence | None


def ensure_task_model(filename: str, model_url: str) -> str:
    """Ensure a MediaPipe .task model exists under app/models and return its path."""
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(model_url, model_path)
    return model_path


def ensure_pose_task_model(model_complexity: int) -> str:
    complexity = int(model_complexity)
    if complexity not in _POSE_MODEL_SPECS:
        complexity = 1
    filename, url = _POSE_MODEL_SPECS[complexity]
    return ensure_task_model(filename, url)


def ensure_hand_task_model() -> str:
    return ensure_task_model("hand_landmarker.task", _HAND_MODEL_URL)


def ensure_face_task_model() -> str:
    return ensure_task_model("face_landmarker.task", _FACE_MODEL_URL)


def landmarks_to_vector(
    landmarks: Sequence | None,
    expected_count: int,
) -> np.ndarray:
    if landmarks is None:
        return np.zeros(expected_count * 3)
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float64,
    ).reshape(-1)


def landmarks_bbox(
    landmarks: Sequence | None,
) -> tuple[float, float, float, float] | None:
    if landmarks is None:
        return None
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (min(xs), min(ys), max(xs), max(ys))


def merge_bboxes(
    b1: tuple[float, float, float, float] | None,
    b2: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    boxes = [bbox for bbox in (b1, b2) if bbox is not None]
    if not boxes:
        return None
    xs = [bbox[0] for bbox in boxes] + [bbox[2] for bbox in boxes]
    ys = [bbox[1] for bbox in boxes] + [bbox[3] for bbox in boxes]
    return (min(xs), min(ys), max(xs), max(ys))


def _frame_to_mp_image(frame_bgr: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _select_hands(result) -> tuple[Sequence | None, Sequence | None]:
    chosen: dict[str, tuple[float, Sequence]] = {}

    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        if not handedness:
            continue
        category = handedness[0]
        label = category.category_name.lower()
        if label not in {"left", "right"}:
            continue
        score = float(getattr(category, "score", 0.0))
        previous = chosen.get(label)
        if previous is None or score > previous[0]:
            chosen[label] = (score, landmarks)

    left = chosen.get("left", (0.0, None))[1]
    right = chosen.get("right", (0.0, None))[1]
    return left, right


class HandVideoLandmarker:
    """
    Stateful hand-only detector for keyframe extraction.

    This uses VIDEO mode so MediaPipe can reuse tracking across frames, which is
    the closest Tasks equivalent to the old Holistic video path with temporal
    smoothing enabled.
    """

    def __init__(self) -> None:
        model_path = ensure_hand_task_model()
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
    ) -> HolisticLikeResult:
        result = self._detector.detect_for_video(
            _frame_to_mp_image(frame_bgr), timestamp_ms
        )
        left_hand, right_hand = _select_hands(result)
        return HolisticLikeResult(
            left_hand_landmarks=left_hand,
            right_hand_landmarks=right_hand,
            pose_landmarks=None,
            face_landmarks=None,
        )

    def close(self) -> None:
        self._detector.close()


class HolisticLikeImageLandmarker:
    """
    Composed Tasks detector that preserves the old landmark pipeline contract.

    The previous landmark extractor used Holistic with static_image_mode=True,
    which means each frame was processed independently. IMAGE mode is the closest
    Tasks equivalent for that behavior.
    """

    def __init__(self, model_complexity: int = 1) -> None:
        pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=ensure_pose_task_model(model_complexity)
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=ensure_hand_task_model()
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        face_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=ensure_face_task_model()
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self._pose = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)
        self._hands = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
        self._face = mp.tasks.vision.FaceLandmarker.create_from_options(face_options)

    def detect(self, frame_bgr: np.ndarray) -> HolisticLikeResult:
        mp_image = _frame_to_mp_image(frame_bgr)
        pose_result = self._pose.detect(mp_image)
        hand_result = self._hands.detect(mp_image)
        face_result = self._face.detect(mp_image)

        left_hand, right_hand = _select_hands(hand_result)
        pose_landmarks = (
            pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
        )
        face_landmarks = (
            face_result.face_landmarks[0] if face_result.face_landmarks else None
        )

        return HolisticLikeResult(
            left_hand_landmarks=left_hand,
            right_hand_landmarks=right_hand,
            pose_landmarks=pose_landmarks,
            face_landmarks=face_landmarks,
        )

    def close(self) -> None:
        self._pose.close()
        self._hands.close()
        self._face.close()
