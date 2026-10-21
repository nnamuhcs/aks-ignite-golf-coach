"""Geometry helpers for pose measurements."""

import math
from typing import Iterable, Tuple

import numpy as np

from .schemas import Landmark


def midpoint(first: Landmark, second: Landmark) -> np.ndarray:
    return (to_vector(first) + to_vector(second)) / 2.0


def to_vector(point: Landmark) -> np.ndarray:
    return np.array([point.x, point.y, point.z], dtype=np.float64)


def angle_degrees(first: Landmark, vertex: Landmark, third: Landmark) -> float:
    left = to_vector(first) - to_vector(vertex)
    right = to_vector(third) - to_vector(vertex)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def horizontal_rotation(first: Landmark, second: Landmark) -> float:
    vector = to_vector(second) - to_vector(first)
    return math.degrees(math.atan2(float(vector[2]), float(vector[0])))


def spine_tilt(shoulder_midpoint: np.ndarray, hip_midpoint: np.ndarray) -> float:
    spine = shoulder_midpoint - hip_midpoint
    vertical = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    denominator = float(np.linalg.norm(spine))
    if denominator == 0:
        return 0.0
    cosine = float(np.clip(np.dot(spine, vertical) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def average_visibility(points: Iterable[Landmark]) -> float:
    values = [point.visibility for point in points]
    return float(sum(values) / len(values)) if values else 0.0


def landmark_tuple(point: Landmark) -> Tuple[float, float, float]:
    return point.x, point.y, point.z
