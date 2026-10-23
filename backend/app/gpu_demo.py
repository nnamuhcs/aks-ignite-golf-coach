"""Manual CUDA demo load, deliberately separate from golf swing analysis.

The API process never touches CUDA for this feature. It launches
``python -m app.gpu_demo`` as a child process that runs real CuPy matrix
multiplications with bounded device memory, a moderate duty cycle and a hard
wall clock deadline. The worker reports only values it measures itself.
"""

import argparse
import importlib.util
import json
import logging
import math
import os
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from .schemas import GpuDemoStatus

LOGGER = logging.getLogger(__name__)

DEMO_LABEL = "GPU demo load: synthetic CUDA matrix math, separate from golf swing analysis"
MAX_DURATION_SECONDS = 120
MIN_DURATION_SECONDS = 5
MAX_MEMORY_MB = 1024
MIN_MEMORY_MB = 16
MAX_DUTY_CYCLE = 0.8
MIN_DUTY_CYCLE = 0.1
DUTY_PERIOD_SECONDS = 0.5
PROGRESS_INTERVAL_SECONDS = 1.0
EXIT_UNAVAILABLE = 3
EXIT_ERROR = 1
BACKEND_DIR = Path(__file__).resolve().parent.parent
ACTIVE_STATES = {"starting", "running", "stopping"}


class GpuDemoError(Exception):
    """Base error for GPU demo control."""


class GpuDemoUnavailable(GpuDemoError):
    """CUDA load cannot run in this image or on this host."""


class GpuDemoConflict(GpuDemoError):
    """A GPU demo run is already active."""


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        LOGGER.warning("Ignoring invalid %s=%r; using default %s", name, raw, default)
        return default
    if not math.isfinite(value):
        LOGGER.warning("Ignoring non-finite %s=%r; using default %s", name, raw, default)
        return default
    return value


def configured_memory_mb() -> int:
    return int(_clamp(_env_float("GPU_DEMO_MEMORY_MB", 256), MIN_MEMORY_MB, MAX_MEMORY_MB))


def configured_duty_cycle() -> float:
    return round(_clamp(_env_float("GPU_DEMO_DUTY_CYCLE", 0.6), MIN_DUTY_CYCLE, MAX_DUTY_CYCLE), 2)


def cupy_installed() -> bool:
    return importlib.util.find_spec("cupy") is not None


def matrix_size_for_budget(memory_mb: int) -> int:
    """Largest multiple of 256 so three float32 N x N matrices fit in the budget."""
    budget_bytes = memory_mb * 1024 * 1024
    size = int((budget_bytes / (3 * 4)) ** 0.5)
    return max(256, min(8192, (size // 256) * 256))


class GpuDemoController:
    """Owns at most one worker process and exposes its real lifecycle state."""

    def __init__(
        self,
        command_factory: Optional[Callable[[float, int, float], List[str]]] = None,
        availability_check: Callable[[], bool] = cupy_installed,
        max_duration_seconds: int = MAX_DURATION_SECONDS,
        memory_mb: Optional[int] = None,
        duty_cycle: Optional[float] = None,
        watchdog_grace_seconds: float = 10.0,
        terminate_timeout_seconds: float = 5.0,
    ) -> None:
        self._command_factory = command_factory or self._worker_command
        self._availability_check = availability_check
        self.max_duration_seconds = int(min(max_duration_seconds, MAX_DURATION_SECONDS))
        self.memory_mb = memory_mb if memory_mb is not None else configured_memory_mb()
        self.duty_cycle = duty_cycle if duty_cycle is not None else configured_duty_cycle()
        self._grace = watchdog_grace_seconds
        self._terminate_timeout = terminate_timeout_seconds
        self._lock = threading.RLock()
        self._process: Optional["subprocess.Popen[str]"] = None
        self._monitor: Optional[threading.Thread] = None
        self._state: Dict[str, Any] = self._blank_state()

    def _blank_state(self) -> Dict[str, Any]:
        return {
            "state": "idle",
            "run_id": None,
            "pid": None,
            "device_name": None,
            "matrix_size": None,
            "allocated_memory_mb": None,
            "requested_duration_seconds": None,
            "started_at": None,
            "started_monotonic": None,
            "ended_monotonic": None,
            "deadline_monotonic": None,
            "iterations": None,
            "gpu_busy_seconds": None,
            "measured_tflops": None,
            "stop_reason": None,
            "stop_requested_monotonic": None,
            "exit_code": None,
            "error": None,
            "worker_finished": None,
            "message": "GPU demo load is idle",
        }

    def _worker_command(self, deadline_epoch: float, memory_mb: int, duty_cycle: float) -> List[str]:
        return [
            sys.executable,
            "-m",
            "app.gpu_demo",
            "--deadline-epoch",
            f"{deadline_epoch:.3f}",
            "--memory-mb",
            str(memory_mb),
            "--duty-cycle",
            str(duty_cycle),
        ]

    def start(self, duration_seconds: Optional[int] = None) -> GpuDemoStatus:
        duration = int(duration_seconds or self.max_duration_seconds)
        if duration < 1 or duration > self.max_duration_seconds:
            raise ValueError(f"Duration must be between 1 and {self.max_duration_seconds} seconds.")
        with self._lock:
            if self._state["state"] in ACTIVE_STATES:
                raise GpuDemoConflict("A GPU demo load is already active. Stop it before starting again.")
            if not self._availability_check():
                raise GpuDemoUnavailable(
                    "CUDA demo load is unavailable: CuPy is not installed in this backend image. "
                    "Use the GPU backend image on a CUDA host."
                )
            deadline_epoch = time.time() + duration
            command = self._command_factory(deadline_epoch, self.memory_mb, self.duty_cycle)
            environment = dict(os.environ, PYTHONUNBUFFERED="1")
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(BACKEND_DIR),
                    env=environment,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
            except OSError as error:
                raise GpuDemoError(f"GPU demo worker could not be launched: {error}") from error

            now = time.monotonic()
            self._state = self._blank_state()
            self._state.update(
                state="starting",
                run_id=str(uuid.uuid4()),
                pid=process.pid,
                requested_duration_seconds=duration,
                started_at=datetime.now(timezone.utc),
                started_monotonic=now,
                deadline_monotonic=now + duration,
                message="Starting CUDA worker process",
            )
            self._process = process
            stderr_tail: Deque[str] = deque(maxlen=20)
            readers = [
                threading.Thread(target=self._read_events, args=(process,), daemon=True),
                threading.Thread(target=self._drain_stderr, args=(process, stderr_tail), daemon=True),
            ]
            for reader in readers:
                reader.start()
            self._monitor = threading.Thread(
                target=self._watch,
                args=(process, readers, stderr_tail),
                name="gpu-demo-monitor",
                daemon=True,
            )
            self._monitor.start()
            LOGGER.info("GPU demo worker %s started for %s seconds", process.pid, duration)
            return self._snapshot()

    def stop(self, reason: str = "user") -> GpuDemoStatus:
        with self._lock:
            process = self._process
            if process is None or self._state["state"] not in ACTIVE_STATES:
                return self._snapshot()
            if self._state["state"] != "stopping":
                self._state.update(
                    state="stopping",
                    stop_reason=reason,
                    stop_requested_monotonic=time.monotonic(),
                    message="Stopping CUDA worker process",
                )
                self._signal(process, signal.SIGTERM)
            return self._snapshot()

    def status(self) -> GpuDemoStatus:
        with self._lock:
            return self._snapshot()

    def shutdown(self, timeout_seconds: Optional[float] = None) -> None:
        """Stop any active worker and wait for it to exit, escalating to SIGKILL."""
        self.stop(reason="shutdown")
        with self._lock:
            monitor = self._monitor
        if monitor is not None:
            monitor.join(timeout_seconds or self._terminate_timeout + 2)
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            self._signal(process, signal.SIGKILL)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                LOGGER.error("GPU demo worker %s did not exit after SIGKILL", process.pid)

    @staticmethod
    def _signal(process: "subprocess.Popen[str]", signum: int) -> None:
        if process.poll() is not None:
            return
        try:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, signum)
            else:
                process.send_signal(signum)
        except ProcessLookupError:
            pass
        except PermissionError:
            process.send_signal(signum)

    def _read_events(self, process: "subprocess.Popen[str]") -> None:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                LOGGER.info("gpu-demo worker: %s", line)
                continue
            if isinstance(event, dict):
                self._apply_event(process, event)

    @staticmethod
    def _drain_stderr(process: "subprocess.Popen[str]", tail: Deque[str]) -> None:
        assert process.stderr is not None
        for line in process.stderr:
            if line.strip():
                tail.append(line.rstrip())

    def _apply_event(self, process: "subprocess.Popen[str]", event: Dict[str, Any]) -> None:
        kind = event.get("event")
        with self._lock:
            if self._process is not process:
                return
            for key in ("device_name", "matrix_size", "allocated_memory_mb", "iterations",
                        "gpu_busy_seconds", "measured_tflops"):
                if key in event and event[key] is not None:
                    self._state[key] = event[key]
            if kind == "ready" and self._state["state"] == "starting":
                self._state.update(state="running", message="CUDA demo load running")
            elif kind == "error":
                self._state["error"] = str(event.get("detail") or "GPU demo worker reported an error.")
            elif kind == "finished":
                self._state["worker_finished"] = event.get("reason")

    def _watch(
        self,
        process: "subprocess.Popen[str]",
        readers: List[threading.Thread],
        stderr_tail: Deque[str],
    ) -> None:
        killed = False
        while process.poll() is None:
            now = time.monotonic()
            with self._lock:
                deadline = self._state["deadline_monotonic"]
                stop_requested = self._state["stop_requested_monotonic"]
            if stop_requested is None and now > deadline + self._grace:
                LOGGER.warning("GPU demo worker %s exceeded its deadline; terminating", process.pid)
                self.stop(reason="watchdog")
            elif (
                stop_requested is not None
                and not killed
                and now - stop_requested > self._terminate_timeout
            ):
                LOGGER.warning("GPU demo worker %s ignored SIGTERM; sending SIGKILL", process.pid)
                self._signal(process, signal.SIGKILL)
                killed = True
            time.sleep(0.1)

        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        for reader in readers:
            reader.join(timeout=2)
        exit_code = process.returncode
        with self._lock:
            if self._process is not process:
                return
            state = self._state
            state.update(exit_code=exit_code, ended_monotonic=time.monotonic(), pid=None)
            if state["stop_reason"] in {"user", "shutdown"}:
                state.update(state="stopped", message="GPU demo load stopped")
            elif state["stop_reason"] == "watchdog":
                state.update(
                    state="stopped",
                    message="GPU demo worker was terminated by the time limit watchdog",
                )
            elif exit_code == 0 and state["worker_finished"] == "time_limit":
                state.update(
                    state="completed",
                    stop_reason="time_limit",
                    message="GPU demo load finished at its time limit",
                )
            elif exit_code == 0 and state["error"] is None:
                state.update(state="completed", message="GPU demo worker exited")
            else:
                if state["error"] is None:
                    detail = stderr_tail[-1] if stderr_tail else "no error output"
                    state["error"] = f"GPU demo worker exited with code {exit_code}: {detail}"
                state.update(state="failed", stop_reason="error", message="GPU demo load failed")
            LOGGER.info("GPU demo worker finished: state=%s exit=%s", state["state"], exit_code)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def _snapshot(self) -> GpuDemoStatus:
        state = self._state
        elapsed = None
        remaining = None
        if state["started_monotonic"] is not None:
            end = state["ended_monotonic"] or time.monotonic()
            elapsed = round(end - state["started_monotonic"], 1)
            if state["state"] in ACTIVE_STATES:
                remaining = round(max(0.0, state["deadline_monotonic"] - time.monotonic()), 1)
        return GpuDemoStatus(
            label=DEMO_LABEL,
            state=state["state"],
            active=state["state"] in ACTIVE_STATES,
            run_id=state["run_id"],
            pid=state["pid"],
            device_name=state["device_name"],
            matrix_size=state["matrix_size"],
            allocated_memory_mb=state["allocated_memory_mb"],
            memory_limit_mb=self.memory_mb,
            duty_cycle=self.duty_cycle,
            max_duration_seconds=self.max_duration_seconds,
            requested_duration_seconds=state["requested_duration_seconds"],
            started_at=state["started_at"],
            elapsed_seconds=elapsed,
            remaining_seconds=remaining,
            iterations=state["iterations"],
            gpu_busy_seconds=state["gpu_busy_seconds"],
            measured_tflops=state["measured_tflops"],
            stop_reason=state["stop_reason"],
            exit_code=state["exit_code"],
            error=state["error"],
            message=state["message"],
        )


def _emit(event: str, **fields: Any) -> None:
    sys.stdout.write(json.dumps({"event": event, **fields}) + "\n")
    sys.stdout.flush()


def _watch_parent(stop: threading.Event) -> None:
    """Stop when the API process closes our stdin or dies."""
    try:
        while sys.stdin.read(4096):
            pass
    except (OSError, ValueError):
        pass
    stop.set()


def run_worker(deadline_epoch: float, memory_mb: int, duty_cycle: float) -> int:
    remaining = deadline_epoch - time.time()
    if remaining > MAX_DURATION_SECONDS + 1:
        _emit("error", kind="invalid", detail="Requested deadline exceeds the 120 second limit.")
        return EXIT_ERROR
    memory_mb = int(_clamp(memory_mb, MIN_MEMORY_MB, MAX_MEMORY_MB))
    duty_cycle = _clamp(duty_cycle, MIN_DUTY_CYCLE, MAX_DUTY_CYCLE)

    stop = threading.Event()

    def request_stop(signum: int, frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        stdin_is_pipe = stat.S_ISFIFO(os.fstat(sys.stdin.fileno()).st_mode)
    except (OSError, ValueError, AttributeError):
        stdin_is_pipe = False
    if stdin_is_pipe:
        threading.Thread(target=_watch_parent, args=(stop,), daemon=True).start()

    try:
        import cupy as cp
    except ImportError as error:
        _emit("error", kind="unavailable",
              detail=f"CUDA demo load is unavailable: CuPy is not installed ({error}).")
        return EXIT_UNAVAILABLE

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as error:
        _emit("error", kind="unavailable", detail=f"CUDA runtime unavailable: {error}")
        return EXIT_UNAVAILABLE
    if device_count < 1:
        _emit("error", kind="unavailable", detail="No CUDA device was detected.")
        return EXIT_UNAVAILABLE

    size = matrix_size_for_budget(memory_mb)
    flops_per_iteration = 2.0 * size ** 3
    a = b = c = None
    try:
        with cp.cuda.Device(0):
            properties = cp.cuda.runtime.getDeviceProperties(0)
            raw_name = properties.get("name", "CUDA GPU")
            device_name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
            required_bytes = 3 * size * size * 4
            try:
                free_bytes, _ = cp.cuda.runtime.memGetInfo()
            except Exception as error:
                raise RuntimeError(f"could not query free GPU memory: {error}") from error
            if free_bytes < required_bytes * 2:
                _emit("error", kind="error",
                      detail=f"Insufficient free GPU memory: need {required_bytes // 2**20} MB "
                             f"with headroom, {free_bytes // 2**20} MB free.")
                return EXIT_ERROR

            cp.random.seed(0)
            a = cp.random.random_sample((size, size), dtype=cp.float32)
            b = cp.random.random_sample((size, size), dtype=cp.float32)
            c = cp.empty((size, size), dtype=cp.float32)
            cp.cuda.Device(0).synchronize()
            allocated = (a.nbytes + b.nbytes + c.nbytes) / 2**20
            _emit("ready", device_name=device_name, matrix_size=size,
                  allocated_memory_mb=round(allocated, 1), iterations=0)

            iterations = 0
            busy_seconds = 0.0
            last_progress = time.monotonic()
            reason = "time_limit"
            while True:
                if stop.is_set():
                    reason = "stopped"
                    break
                if time.time() >= deadline_epoch:
                    break
                period_start = time.monotonic()
                busy_target = DUTY_PERIOD_SECONDS * duty_cycle
                while time.monotonic() - period_start < busy_target and not stop.is_set():
                    started = time.monotonic()
                    cp.matmul(a, b, out=c)
                    cp.cuda.Device(0).synchronize()
                    busy_seconds += time.monotonic() - started
                    iterations += 1
                    if time.time() >= deadline_epoch:
                        break
                idle = DUTY_PERIOD_SECONDS - (time.monotonic() - period_start)
                if idle > 0:
                    stop.wait(min(idle, max(0.0, deadline_epoch - time.time())))
                if time.monotonic() - last_progress >= PROGRESS_INTERVAL_SECONDS:
                    last_progress = time.monotonic()
                    tflops = flops_per_iteration * iterations / busy_seconds / 1e12 if busy_seconds else None
                    _emit("progress", iterations=iterations, gpu_busy_seconds=round(busy_seconds, 2),
                          measured_tflops=round(tflops, 3) if tflops is not None else None)

            checksum = float(c[0, 0]) if iterations else None
            tflops = flops_per_iteration * iterations / busy_seconds / 1e12 if busy_seconds else None
            _emit("finished", reason=reason, iterations=iterations,
                  gpu_busy_seconds=round(busy_seconds, 2),
                  measured_tflops=round(tflops, 3) if tflops is not None else None,
                  checksum=checksum)
            return 0
    except Exception as error:
        _emit("error", kind="error", detail=f"CUDA demo load failed: {error}")
        return EXIT_ERROR
    finally:
        del a, b, c
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=DEMO_LABEL)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--memory-mb", type=int, default=256)
    parser.add_argument("--duty-cycle", type=float, default=0.6)
    arguments = parser.parse_args(argv)
    return run_worker(arguments.deadline_epoch, arguments.memory_mb, arguments.duty_cycle)


if __name__ == "__main__":
    sys.exit(main())
