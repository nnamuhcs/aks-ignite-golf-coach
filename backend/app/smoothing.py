"""Adaptive temporal smoothing for 3D pose landmarks."""

from typing import List

import numpy as np

from .schemas import Landmark, PoseFrame


def _adaptive_pass(values: np.ndarray) -> np.ndarray:
    output = values.copy()
    for index in range(1, len(values)):
        motion = np.linalg.norm(values[index] - output[index - 1], axis=1, keepdims=True)
        alpha = np.clip(0.34 + motion * 4.5, 0.34, 0.86)
        output[index] = alpha * values[index] + (1.0 - alpha) * output[index - 1]
    return output


def smooth_world_landmarks(frames: List[PoseFrame]) -> List[PoseFrame]:
    """Apply forward and backward adaptive smoothing without shifting phase timing."""
    if len(frames) < 3:
        return frames

    values = np.array(
        [
            [[point.x, point.y, point.z] for point in frame.world_landmarks]
            for frame in frames
        ],
        dtype=np.float64,
    )
    forward = _adaptive_pass(values)
    backward = _adaptive_pass(values[::-1])[::-1]
    smoothed = (forward + backward) / 2.0

    output = []
    for frame_index, frame in enumerate(frames):
        landmarks = [
            Landmark(
                x=float(smoothed[frame_index, point_index, 0]),
                y=float(smoothed[frame_index, point_index, 1]),
                z=float(smoothed[frame_index, point_index, 2]),
                visibility=point.visibility,
            )
            for point_index, point in enumerate(frame.world_landmarks)
        ]
        output.append(frame.model_copy(update={"world_landmarks": landmarks}))
    return output
