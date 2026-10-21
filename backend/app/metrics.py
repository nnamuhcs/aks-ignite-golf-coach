"""Biomechanical metrics computed from MediaPipe world landmarks."""

from typing import Dict, List, Optional

import numpy as np

from .geometry import angle_degrees, horizontal_rotation, midpoint, spine_tilt
from .schemas import ComparisonMetric, PoseFrame, TrajectoryPoint

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

METRIC_UNITS = {
    "shoulder_turn": "degrees",
    "hip_turn": "degrees",
    "spine_tilt": "degrees",
    "lead_arm_extension": "degrees",
    "knee_flex": "degrees",
}


def calculate_phase_metrics(frame: PoseFrame, handedness: str) -> Dict[str, float]:
    points = frame.world_landmarks
    shoulder_mid = midpoint(points[LEFT_SHOULDER], points[RIGHT_SHOULDER])
    hip_mid = midpoint(points[LEFT_HIP], points[RIGHT_HIP])

    if handedness == "left":
        lead_shoulder, lead_elbow, lead_wrist = RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
    else:
        lead_shoulder, lead_elbow, lead_wrist = LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST

    left_knee_flex = angle_degrees(points[LEFT_HIP], points[LEFT_KNEE], points[LEFT_ANKLE])
    right_knee_flex = angle_degrees(points[RIGHT_HIP], points[RIGHT_KNEE], points[RIGHT_ANKLE])

    return {
        "shoulder_turn": round(abs(horizontal_rotation(points[LEFT_SHOULDER], points[RIGHT_SHOULDER])), 1),
        "hip_turn": round(abs(horizontal_rotation(points[LEFT_HIP], points[RIGHT_HIP])), 1),
        "spine_tilt": round(spine_tilt(shoulder_mid, hip_mid), 1),
        "lead_arm_extension": round(
            angle_degrees(points[lead_shoulder], points[lead_elbow], points[lead_wrist]),
            1,
        ),
        "knee_flex": round((left_knee_flex + right_knee_flex) / 2.0, 1),
    }


def calculate_trajectory(frames: List[PoseFrame]) -> List[TrajectoryPoint]:
    trajectory: List[TrajectoryPoint] = []
    for frame in frames:
        left = frame.world_landmarks[LEFT_WRIST]
        right = frame.world_landmarks[RIGHT_WRIST]
        trajectory.append(
            TrajectoryPoint(
                timestamp_seconds=round(frame.timestamp_seconds, 3),
                x=round((left.x + right.x) / 2.0, 4),
                y=round((left.y + right.y) / 2.0, 4),
                z=round((left.z + right.z) / 2.0, 4),
            )
        )
    return trajectory


def calculate_tempo(
    frames: List[PoseFrame],
    top_index: int,
    impact_index: int,
) -> Optional[float]:
    backswing = frames[top_index].timestamp_seconds - frames[0].timestamp_seconds
    downswing = frames[impact_index].timestamp_seconds - frames[top_index].timestamp_seconds
    if backswing <= 0 or downswing <= 0:
        return None
    return round(backswing / downswing, 2)


def compare_metrics(
    athlete: Dict[str, Dict[str, float]],
    reference: Dict[str, Dict[str, float]],
) -> List[ComparisonMetric]:
    comparison: List[ComparisonMetric] = []
    for phase, athlete_metrics in athlete.items():
        reference_metrics = reference.get(phase)
        if not reference_metrics:
            continue
        for metric, athlete_value in athlete_metrics.items():
            reference_value = reference_metrics[metric]
            comparison.append(
                ComparisonMetric(
                    metric=f"{phase}.{metric}",
                    athlete_value=athlete_value,
                    reference_value=reference_value,
                    difference=round(athlete_value - reference_value, 1),
                    unit=METRIC_UNITS[metric],
                )
            )
    return comparison


def reference_score(comparison: List[ComparisonMetric]) -> Optional[float]:
    if not comparison:
        return None
    normalized_errors = []
    for item in comparison:
        scale = 45.0 if item.unit == "degrees" else max(abs(item.reference_value), 1.0)
        normalized_errors.append(min(abs(item.difference) / scale, 1.0))
    return round(max(0.0, 100.0 * (1.0 - float(np.mean(normalized_errors)))), 1)
