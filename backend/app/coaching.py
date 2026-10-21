"""Evidence-based coaching rules over measured swing metrics."""

from typing import Dict, List, Optional

from .schemas import CoachingInsight, ComparisonMetric


def build_coaching(
    phase_metrics: Dict[str, Dict[str, float]],
    tempo_ratio: Optional[float],
    comparison: List[ComparisonMetric],
) -> List[CoachingInsight]:
    insights: List[CoachingInsight] = []
    top = phase_metrics["top"]
    impact = phase_metrics["impact"]
    address = phase_metrics["address"]

    if top["lead_arm_extension"] < 145:
        insights.append(
            CoachingInsight(
                phase="top",
                title="Create more width at the top",
                detail="Keep the lead arm longer through the backswing to preserve swing radius.",
                severity="focus",
                evidence=f"Lead arm angle measured {top['lead_arm_extension']:.1f} degrees.",
            )
        )
    else:
        insights.append(
            CoachingInsight(
                phase="top",
                title="Strong backswing width",
                detail="The lead arm remains extended at the top of the swing.",
                severity="strength",
                evidence=f"Lead arm angle measured {top['lead_arm_extension']:.1f} degrees.",
            )
        )

    rotation_gap = top["shoulder_turn"] - top["hip_turn"]
    insights.append(
        CoachingInsight(
            phase="top",
            title="Measured upper and lower body separation",
            detail="Use this rotation gap as a repeatability baseline across future swings.",
            severity="information",
            evidence=f"Shoulder to hip rotation gap measured {rotation_gap:.1f} degrees.",
        )
    )

    spine_change = impact["spine_tilt"] - address["spine_tilt"]
    if abs(spine_change) > 18:
        insights.append(
            CoachingInsight(
                phase="impact",
                title="Stabilize posture through impact",
                detail="Reduce the change in torso inclination between address and impact.",
                severity="focus",
                evidence=f"Spine tilt changed {spine_change:+.1f} degrees.",
            )
        )

    if tempo_ratio is not None:
        severity = "strength" if 2.4 <= tempo_ratio <= 3.6 else "focus"
        insights.append(
            CoachingInsight(
                phase="downswing",
                title="Backswing to downswing tempo",
                detail=(
                    "Tempo is within the commonly observed three-to-one range."
                    if severity == "strength"
                    else "Use a smoother transition to move the ratio closer to three-to-one."
                ),
                severity=severity,
                evidence=f"Measured tempo ratio is {tempo_ratio:.2f}:1.",
            )
        )

    if comparison:
        largest = max(comparison, key=lambda item: abs(item.difference))
        phase, metric = largest.metric.split(".", 1)
        insights.append(
            CoachingInsight(
                phase=phase,
                title="Largest reference difference",
                detail=f"Prioritize {metric.replace('_', ' ')} when matching the reference swing.",
                severity="focus",
                evidence=(
                    f"Athlete {largest.athlete_value:.1f}, reference "
                    f"{largest.reference_value:.1f} {largest.unit}."
                ),
            )
        )

    return insights
