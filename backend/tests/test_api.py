"""API contract tests that do not fabricate inference data."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import UPLOAD_DIR
from app.main import app
from app.pose_estimator import PoseEstimationError, read_video_display_rotation

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_missing_analysis_returns_not_found() -> None:
    response = client.get("/api/analyses/not-a-real-analysis")
    assert response.status_code == 404


def test_invalid_upload_type_is_rejected() -> None:
    directories_before = set(UPLOAD_DIR.iterdir())
    response = client.post(
        "/api/analyses",
        files={"athlete_video": ("notes.txt", b"not a video", "text/plain")},
        data={"handedness": "right"},
    )
    assert response.status_code == 400
    assert set(UPLOAD_DIR.iterdir()) == directories_before


def test_missing_video_does_not_claim_a_display_rotation(tmp_path: Path) -> None:
    with pytest.raises(PoseEstimationError, match="display rotation could not be read"):
        read_video_display_rotation(tmp_path / "missing.mp4")
