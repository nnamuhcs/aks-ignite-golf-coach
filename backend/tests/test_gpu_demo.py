"""Process lifecycle tests for the manual GPU demo load.

These run on CPU only hosts. Real CUDA math is never simulated: lifecycle tests
use small stand-in child processes that emit lifecycle events only, with no
device names or performance metrics, so the controller's process handling is
exercised with real subprocesses, signals and exit codes.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List

import pytest
from fastapi.testclient import TestClient

from app import main
from app.gpu_demo import (
    GpuDemoConflict,
    GpuDemoController,
    GpuDemoUnavailable,
    configured_duty_cycle,
    configured_memory_mb,
    cupy_installed,
    matrix_size_for_budget,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent
requires_no_cupy = pytest.mark.skipif(cupy_installed(), reason="CuPy is installed on this host")

RUN_UNTIL_SIGTERM = """
import json, signal, sys, time
stop = []
signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
print(json.dumps({"event": "ready"}), flush=True)
while not stop:
    time.sleep(0.05)
print(json.dumps({"event": "finished", "reason": "stopped"}), flush=True)
"""

HONOR_DEADLINE = """
import json, sys, time
deadline = float(sys.argv[1])
print(json.dumps({"event": "ready"}), flush=True)
while time.time() < deadline:
    time.sleep(0.05)
print(json.dumps({"event": "finished", "reason": "time_limit"}), flush=True)
"""

IGNORE_SIGTERM = """
import json, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print(json.dumps({"event": "ready"}), flush=True)
while True:
    time.sleep(0.05)
"""

REPORT_ERROR = """
import json, sys
print(json.dumps({"event": "error", "kind": "unavailable",
                  "detail": "CUDA runtime unavailable: no CUDA-capable device"}), flush=True)
sys.exit(3)
"""

CRASH = """
import sys
sys.stderr.write("worker crashed before reporting\\n")
sys.exit(7)
"""


def stub(script: str) -> Callable[[float, int, float], List[str]]:
    return lambda deadline, memory_mb, duty: [sys.executable, "-c", script, f"{deadline:.3f}"]


def make_controller(script: str, **kwargs: object) -> GpuDemoController:
    options = dict(
        availability_check=lambda: True,
        memory_mb=64,
        duty_cycle=0.5,
        watchdog_grace_seconds=0.5,
        terminate_timeout_seconds=0.5,
    )
    options.update(kwargs)
    return GpuDemoController(command_factory=stub(script), **options)


def wait_for(controller: GpuDemoController, states: set, timeout: float = 10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        status = controller.status()
        if status.state in states:
            return status
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {states}; last state {controller.status().state}")


def test_matrix_budget_stays_within_memory_limit() -> None:
    for memory_mb in (16, 64, 256, 1024):
        size = matrix_size_for_budget(memory_mb)
        assert size % 256 == 0
        assert 3 * size * size * 4 <= memory_mb * 1024 * 1024


@pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "-inf", "Infinity", "not-a-number"])
def test_invalid_env_values_fall_back_to_defaults(
    raw: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("GPU_DEMO_MEMORY_MB", raw)
    monkeypatch.setenv("GPU_DEMO_DUTY_CYCLE", raw)
    with caplog.at_level("WARNING", logger="app.gpu_demo"):
        assert configured_memory_mb() == 256
        assert configured_duty_cycle() == 0.6
    messages = [record.getMessage() for record in caplog.records]
    assert any("GPU_DEMO_MEMORY_MB" in message for message in messages)
    assert any("GPU_DEMO_DUTY_CYCLE" in message for message in messages)


def test_finite_env_values_are_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPU_DEMO_MEMORY_MB", "999999")
    monkeypatch.setenv("GPU_DEMO_DUTY_CYCLE", "0.01")
    assert configured_memory_mb() == 1024
    assert configured_duty_cycle() == 0.1


MEMINFO_FAILS_CUPY = """
class runtime:
    @staticmethod
    def getDeviceCount():
        return 1

    @staticmethod
    def getDeviceProperties(index):
        return {"name": b"unused"}

    @staticmethod
    def memGetInfo():
        raise RuntimeError("cudaErrorNotSupported")


class Device:
    def __init__(self, index):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
"""


def test_worker_reports_memory_query_failure(tmp_path: Path) -> None:
    # Error path only: the stand-in module raises before any CUDA work or metrics exist.
    package = tmp_path / "cupy"
    (package / "cuda").mkdir(parents=True)
    (package / "__init__.py").write_text("from . import cuda\n", encoding="utf-8")
    (package / "cuda" / "__init__.py").write_text(MEMINFO_FAILS_CUPY, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "app.gpu_demo", "--deadline-epoch", str(time.time() + 5)],
        cwd=str(BACKEND_DIR),
        env=dict(os.environ, PYTHONPATH=str(tmp_path)),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    events = [json.loads(line) for line in completed.stdout.strip().splitlines()]
    assert completed.returncode == 1
    assert [event["event"] for event in events] == ["error"]
    assert events[0]["detail"] == (
        "CUDA demo load failed: could not query free GPU memory: cudaErrorNotSupported"
    )


def test_initial_status_is_idle_and_labeled() -> None:
    status = make_controller(RUN_UNTIL_SIGTERM).status()
    assert status.state == "idle"
    assert not status.active
    assert "separate from golf swing analysis" in status.label
    assert status.max_duration_seconds == 120
    assert status.iterations is None and status.measured_tflops is None


@requires_no_cupy
def test_real_worker_reports_missing_cupy() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "app.gpu_demo", "--deadline-epoch", str(time.time() + 5)],
        cwd=str(BACKEND_DIR),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 3
    event = json.loads(completed.stdout.strip().splitlines()[-1])
    assert event["event"] == "error"
    assert event["kind"] == "unavailable"
    assert "CuPy is not installed" in event["detail"]


def test_real_worker_rejects_deadline_beyond_limit() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "app.gpu_demo", "--deadline-epoch", str(time.time() + 600)],
        cwd=str(BACKEND_DIR),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 1
    assert "120 second limit" in json.loads(completed.stdout.strip())["detail"]


@requires_no_cupy
def test_controller_refuses_start_without_cupy() -> None:
    controller = GpuDemoController()
    with pytest.raises(GpuDemoUnavailable, match="CuPy is not installed"):
        controller.start(10)
    assert controller.status().state == "idle"


@requires_no_cupy
def test_real_worker_failure_surfaces_through_controller() -> None:
    controller = GpuDemoController(availability_check=lambda: True)
    status = controller.start(10)
    assert status.state == "starting"
    final = wait_for(controller, {"failed"}, timeout=30)
    assert final.exit_code == 3
    assert "CuPy is not installed" in (final.error or "")
    assert final.device_name is None and final.iterations is None


def test_start_stop_lifecycle_and_duplicates() -> None:
    controller = make_controller(RUN_UNTIL_SIGTERM)
    started = controller.start(30)
    assert started.state == "starting"
    assert started.active and started.pid
    running = wait_for(controller, {"running"})
    assert running.remaining_seconds is not None and running.remaining_seconds <= 30
    with pytest.raises(GpuDemoConflict):
        controller.start(30)
    stopping = controller.stop()
    assert stopping.state in {"stopping", "stopped"}
    assert controller.stop().state in {"stopping", "stopped"}
    final = wait_for(controller, {"stopped"})
    assert final.stop_reason == "user"
    assert final.exit_code == 0
    assert not final.active and final.pid is None
    assert final.device_name is None and final.measured_tflops is None
    assert controller.stop().state == "stopped"

    restarted = controller.start(30)
    assert restarted.run_id != final.run_id
    controller.shutdown()
    assert controller.status().state == "stopped"


def test_stop_when_idle_is_noop() -> None:
    controller = make_controller(RUN_UNTIL_SIGTERM)
    assert controller.stop().state == "idle"


def test_rejects_duration_above_limit() -> None:
    controller = make_controller(RUN_UNTIL_SIGTERM)
    with pytest.raises(ValueError):
        controller.start(121)
    assert controller.status().state == "idle"


def test_worker_time_limit_completes() -> None:
    controller = make_controller(HONOR_DEADLINE)
    controller.start(1)
    final = wait_for(controller, {"completed"})
    assert final.stop_reason == "time_limit"
    assert final.exit_code == 0
    assert 0.9 <= (final.elapsed_seconds or 0) < 5


def test_watchdog_kills_worker_that_ignores_deadline_and_sigterm() -> None:
    controller = make_controller(IGNORE_SIGTERM)
    process_started = time.monotonic()
    controller.start(1)
    wait_for(controller, {"running"})
    final = wait_for(controller, {"stopped"}, timeout=15)
    assert final.stop_reason == "watchdog"
    assert final.exit_code == -9
    assert time.monotonic() - process_started < 10


def test_user_stop_escalates_to_sigkill() -> None:
    controller = make_controller(IGNORE_SIGTERM)
    controller.start(60)
    wait_for(controller, {"running"})
    controller.stop()
    final = wait_for(controller, {"stopped"})
    assert final.exit_code == -9
    assert final.stop_reason == "user"


def test_worker_error_event_marks_failed() -> None:
    controller = make_controller(REPORT_ERROR)
    controller.start(10)
    final = wait_for(controller, {"failed"})
    assert final.exit_code == 3
    assert final.error == "CUDA runtime unavailable: no CUDA-capable device"
    assert final.stop_reason == "error"


def test_worker_crash_reports_stderr() -> None:
    controller = make_controller(CRASH)
    controller.start(10)
    final = wait_for(controller, {"failed"})
    assert final.exit_code == 7
    assert "worker crashed before reporting" in (final.error or "")


def test_shutdown_terminates_active_worker() -> None:
    controller = make_controller(IGNORE_SIGTERM)
    status = controller.start(60)
    wait_for(controller, {"running"})
    controller.shutdown(timeout_seconds=5)
    final = controller.status()
    assert final.state == "stopped"
    assert final.stop_reason == "shutdown"
    with pytest.raises(ProcessLookupError):
        os.kill(status.pid or 0, 0)


def test_api_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = make_controller(RUN_UNTIL_SIGTERM)
    monkeypatch.setattr(main, "GPU_DEMO", controller)
    client = TestClient(main.app)
    try:
        assert client.get("/api/gpu-demo").json()["state"] == "idle"
        assert client.post("/api/gpu-demo/start", json={"duration_seconds": 121}).status_code == 422
        response = client.post("/api/gpu-demo/start", json={"duration_seconds": 30})
        assert response.status_code == 202
        assert response.json()["state"] == "starting"
        wait_for(controller, {"running"})
        duplicate = client.post("/api/gpu-demo/start")
        assert duplicate.status_code == 409
        assert "already active" in duplicate.json()["detail"]
        assert client.get("/api/gpu-demo/status").json()["state"] == "running"
        assert client.post("/api/gpu-demo/stop").status_code == 200
        wait_for(controller, {"stopped"})
        assert client.post("/api/gpu-demo/stop").json()["state"] == "stopped"
    finally:
        controller.shutdown()


def test_api_default_start_uses_full_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = make_controller(RUN_UNTIL_SIGTERM)
    monkeypatch.setattr(main, "GPU_DEMO", controller)
    try:
        response = TestClient(main.app).post("/api/gpu-demo/start")
        assert response.status_code == 202
        assert response.json()["requested_duration_seconds"] == 120
    finally:
        controller.shutdown()


@requires_no_cupy
def test_api_returns_503_without_cupy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "GPU_DEMO", GpuDemoController())
    response = TestClient(main.app).post("/api/gpu-demo/start", json={"duration_seconds": 10})
    assert response.status_code == 503
    assert "CuPy is not installed" in response.json()["detail"]


def test_lifespan_shutdown_stops_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = make_controller(RUN_UNTIL_SIGTERM)
    monkeypatch.setattr(main, "GPU_DEMO", controller)
    with TestClient(main.app) as client:
        client.post("/api/gpu-demo/start", json={"duration_seconds": 30})
        wait_for(controller, {"running"})
    final = controller.status()
    assert final.state == "stopped"
    assert final.stop_reason == "shutdown"
