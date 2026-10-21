"""Data-driven golf swing phase detection."""

from typing import Dict, List, Optional

import numpy as np

from .config import SWING_PHASES
from .schemas import PoseFrame

LEFT_WRIST = 15
RIGHT_WRIST = 16


def _wrist_center(frame: PoseFrame) -> np.ndarray:
    left = frame.image_landmarks[LEFT_WRIST]
    right = frame.image_landmarks[RIGHT_WRIST]
    return np.array([(left.x + right.x) / 2.0, (left.y + right.y) / 2.0], dtype=np.float64)


def detect_phase_indices(
    frames: List[PoseFrame],
    motion_peak_timestamp: Optional[float] = None,
) -> Dict[str, int]:
    """Locate eight ordered swing phases from wrist elevation and velocity."""
    count = len(frames)
    wrist_positions = np.stack([_wrist_center(frame) for frame in frames])
    wrist_velocity = np.linalg.norm(np.diff(wrist_positions, axis=0), axis=1)

    search_start = max(1, int(count * 0.12))
    search_end = max(search_start + 1, int(count * 0.68))
    top_index = search_start + int(np.argmin(wrist_positions[search_start:search_end, 1]))

    impact_start = min(count - 2, top_index + 1)
    impact_end = max(impact_start + 1, int(count * 0.9))
    impact_end = min(impact_end, count - 1)
    velocity_window = wrist_velocity[impact_start:impact_end]
    impact_index = (
        impact_start + int(np.argmax(velocity_window))
        if len(velocity_window)
        else min(count - 2, top_index + 1)
    )
    if motion_peak_timestamp is not None:
        motion_index = min(
            range(count),
            key=lambda index: abs(frames[index].timestamp_seconds - motion_peak_timestamp),
        )
        if top_index < motion_index < count - 1:
            impact_index = round(impact_index * 0.7 + motion_index * 0.3)

    raw = [
        0,
        max(1, int(top_index * 0.35)),
        max(2, int(top_index * 0.72)),
        top_index,
        top_index + max(1, (impact_index - top_index) // 2),
        impact_index,
        impact_index + max(1, (count - 1 - impact_index) // 2),
        count - 1,
    ]

    ordered: List[int] = []
    for position, candidate in enumerate(raw):
        minimum = ordered[-1] + 1 if ordered else 0
        maximum = count - (len(raw) - position)
        ordered.append(max(minimum, min(candidate, maximum)))

    return dict(zip(SWING_PHASES, ordered))


def phase_confidence(frame: PoseFrame) -> float:
    core_indices = (11, 12, 23, 24, 25, 26, 27, 28)
    visibility = [frame.image_landmarks[index].visibility for index in core_indices]
    return float(np.clip(np.mean(visibility), 0.0, 1.0))
