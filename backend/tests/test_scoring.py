from app.scoring import phase_biomechanics_score


def test_phase_score_rewards_values_near_phase_targets():
    ideal = {
        "shoulder_turn": 80.0,
        "hip_turn": 40.0,
        "spine_tilt": 35.0,
        "lead_arm_extension": 155.0,
        "knee_flex": 145.0,
    }
    outside_targets = {
        "shoulder_turn": 20.0,
        "hip_turn": 5.0,
        "spine_tilt": 75.0,
        "lead_arm_extension": 105.0,
        "knee_flex": 175.0,
    }

    ideal_score = phase_biomechanics_score("top", ideal, 0.9)
    outside_score = phase_biomechanics_score("top", outside_targets, 0.9)

    assert ideal_score > 95.0
    assert outside_score < 65.0


def test_phase_score_varies_within_previously_flat_ranges():
    centered = {
        "shoulder_turn": 45.0,
        "hip_turn": 30.0,
        "spine_tilt": 35.0,
        "lead_arm_extension": 155.0,
        "knee_flex": 145.0,
    }
    offset = {
        "shoulder_turn": 75.0,
        "hip_turn": 55.0,
        "spine_tilt": 65.0,
        "lead_arm_extension": 130.0,
        "knee_flex": 170.0,
    }

    centered_score = phase_biomechanics_score("impact", centered, 0.85)
    offset_score = phase_biomechanics_score("impact", offset, 0.85)

    assert centered_score > offset_score
    assert centered_score - offset_score > 20.0
