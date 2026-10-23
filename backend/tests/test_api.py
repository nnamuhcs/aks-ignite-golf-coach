"""API contract tests that do not fabricate inference data."""

from pathlib import Path
import subprocess

import pytest
from fastapi.testclient import TestClient

from app.config import UPLOAD_DIR
from app.main import app
from app.platform import _gpu_status
from app.pose_estimator import PoseEstimationError, read_video_display_rotation

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_thor_telemetry_preserves_unreported_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    # Captured from the Thor host's nvidia-smi output.
    output = "NVIDIA Thor, 595.78, [N/A], [N/A], 2\n"
    monkeypatch.setattr(
        "app.platform.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, output, ""),
    )
    status = _gpu_status()
    assert status.available
    assert status.name == "NVIDIA Thor"
    assert status.driver_version == "595.78"
    assert status.memory_total_mb is None
    assert status.memory_used_mb is None
    assert status.utilization_percent == 2


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
