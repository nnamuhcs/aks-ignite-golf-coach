"""Transparent, brand-neutral biomechanics scoring for the demo experience."""

import math
from typing import Dict, List, Optional, Tuple

from .schemas import PhaseResult

PHASE_WEIGHTS = {
    "address": 0.10,
    "takeaway": 0.10,
    "backswing": 0.12,
    "top": 0.15,
    "downswing": 0.15,
    "impact": 0.18,
    "follow_through": 0.10,
    "finish": 0.10,
}

MetricTarget = Tuple[str, float, float, float]

PHASE_TARGETS: Dict[str, List[MetricTarget]] = {
    "address": [
        ("spine_tilt", 35.0, 14.0, 1.2),
        ("lead_arm_extension", 165.0, 16.0, 1.0),
        ("knee_flex", 155.0, 14.0, 1.0),
    ],
    "takeaway": [
        ("shoulder_turn", 25.0, 16.0, 1.2),
        ("hip_turn", 12.0, 12.0, 0.8),
        ("spine_tilt", 35.0, 16.0, 1.0),
        ("lead_arm_extension", 160.0, 18.0, 1.0),
        ("knee_flex", 155.0, 16.0, 0.8),
    ],
    "backswing": [
        ("shoulder_turn", 55.0, 20.0, 1.2),
        ("hip_turn", 28.0, 16.0, 0.8),
        ("spine_tilt", 35.0, 17.0, 1.0),
        ("lead_arm_extension", 155.0, 20.0, 1.0),
        ("knee_flex", 150.0, 18.0, 0.8),
    ],
    "top": [
        ("shoulder_turn", 80.0, 20.0, 1.3),
        ("hip_turn", 40.0, 17.0, 1.0),
        ("spine_tilt", 35.0, 17.0, 1.0),
        ("lead_arm_extension", 155.0, 18.0, 1.0),
        ("knee_flex", 145.0, 18.0, 0.7),
    ],
    "downswing": [
        ("shoulder_turn", 65.0, 22.0, 1.1),
        ("hip_turn", 35.0, 17.0, 1.0),
        ("spine_tilt", 35.0, 17.0, 1.0),
        ("lead_arm_extension", 150.0, 18.0, 1.0),
        ("knee_flex", 145.0, 17.0, 0.9),
    ],
    "impact": [
        ("shoulder_turn", 45.0, 25.0, 0.8),
        ("hip_turn", 30.0, 18.0, 1.0),
        ("spine_tilt", 35.0, 18.0, 1.0),
        ("lead_arm_extension", 155.0, 18.0, 1.2),
        ("knee_flex", 145.0, 17.0, 1.0),
    ],
    "follow_through": [
        ("shoulder_turn", 45.0, 24.0, 0.8),
        ("hip_turn", 35.0, 20.0, 0.8),
        ("spine_tilt", 28.0, 18.0, 1.0),
        ("lead_arm_extension", 140.0, 22.0, 1.1),
        ("knee_flex", 155.0, 18.0, 0.8),
    ],
    "finish": [
        ("spine_tilt", 22.0, 16.0, 1.0),
        ("lead_arm_extension", 145.0, 22.0, 1.0),
        ("knee_flex", 165.0, 15.0, 1.0),
    ],
}


def _target_score(value: float, target: float, tolerance: float) -> float:
    distance = (value - target) / tolerance
    return 35.0 + 65.0 * math.exp(-0.5 * distance**2)


def phase_biomechanics_score(
    phase: str,
    metrics: Dict[str, float],
    pose_confidence: float,
) -> float:
    weighted_scores = [
        (_target_score(metrics[metric], target, tolerance), weight)
        for metric, target, tolerance, weight in PHASE_TARGETS[phase]
    ]
    biomechanics_score = sum(score * weight for score, weight in weighted_scores) / sum(
        weight for _, weight in weighted_scores
    )
    tracking_score = max(0.0, min(100.0, pose_confidence * 100.0))
    return round(biomechanics_score * 0.92 + tracking_score * 0.08, 1)


def overall_biomechanics_score(
    phases: List[PhaseResult],
    tempo_ratio: Optional[float],
) -> float:
    weighted_score = sum(phase.score * PHASE_WEIGHTS[phase.name] for phase in phases)
    if tempo_ratio is None:
        return round(weighted_score, 1)
    tempo_score = 100.0 * math.exp(-0.5 * ((tempo_ratio - 3.0) / 1.4) ** 2)
    return round(weighted_score * 0.85 + tempo_score * 0.15, 1)
