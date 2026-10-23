"""FastAPI entry point for the AKS Ignite Golf Coach."""

import json
import logging
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import ASSETS_DIR, MAX_UPLOAD_SIZE_MB, RESULTS_DIR, UPLOAD_DIR
from .gpu_demo import GpuDemoConflict, GpuDemoController, GpuDemoError, GpuDemoUnavailable
from .pipeline import run_analysis
from .platform import get_kubernetes_inventory, get_platform_status
from .pose_estimator import PoseEstimationError, read_video_display_rotation
from .schemas import (
    AnalysisResult,
    GpuDemoStartRequest,
    GpuDemoStatus,
    JobStatus,
    KubernetesInventory,
    PlatformStatus,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

GPU_DEMO = GpuDemoController()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        GPU_DEMO.shutdown()


app = FastAPI(title="AKS Ignite Golf Coach API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

JOBS: Dict[str, JobStatus] = {}
JOBS_LOCK = threading.Lock()
ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".webm"}


def _set_job(analysis_id: str, **changes: object) -> None:
    with JOBS_LOCK:
        current = JOBS[analysis_id]
        JOBS[analysis_id] = current.model_copy(update=changes)


def _save_upload(upload: UploadFile, destination: Path) -> None:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Upload a supported video file.")

    with destination.open("wb") as output:
        shutil.copyfileobj(upload.file, output)
    if destination.stat().st_size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"Video exceeds the {MAX_UPLOAD_SIZE_MB} MB upload limit.",
        )


def _process_job(
    analysis_id: str,
    athlete_path: Path,
    athlete_filename: str,
    handedness: str,
    reference_path: Optional[Path],
    reference_filename: Optional[str],
) -> None:
    try:
        _set_job(analysis_id, status="analyzing", progress=5, message="Preparing video analysis")
        result = run_analysis(
            analysis_id=analysis_id,
            athlete_path=athlete_path,
            athlete_filename=athlete_filename,
            handedness=handedness,
            reference_path=reference_path,
            reference_filename=reference_filename,
            progress_callback=lambda progress, message: _set_job(
                analysis_id,
                progress=progress,
                message=message,
            ),
        )
        result_path = RESULTS_DIR / f"{analysis_id}.json"
        result_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        _set_job(
            analysis_id,
            status="completed",
            progress=100,
            message="Analysis complete",
            result_url=f"/api/analyses/{analysis_id}/result",
        )
    except Exception as error:
        LOGGER.exception("Analysis %s failed", analysis_id)
        _set_job(
            analysis_id,
            status="failed",
            progress=100,
            message="Analysis failed",
            error=str(error),
        )


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/api/platform", response_model=PlatformStatus)
def platform_status() -> PlatformStatus:
    return get_platform_status()


@app.get("/api/kubernetes", response_model=KubernetesInventory)
def kubernetes_inventory() -> KubernetesInventory:
    return get_kubernetes_inventory()


@app.get("/api/gpu-demo", response_model=GpuDemoStatus)
@app.get("/api/gpu-demo/status", response_model=GpuDemoStatus)
def gpu_demo_status() -> GpuDemoStatus:
    return GPU_DEMO.status()


@app.post("/api/gpu-demo/start", response_model=GpuDemoStatus, status_code=202)
def start_gpu_demo(request: Optional[GpuDemoStartRequest] = None) -> GpuDemoStatus:
    duration = request.duration_seconds if request else None
    try:
        return GPU_DEMO.start(duration)
    except GpuDemoConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except GpuDemoUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (GpuDemoError, ValueError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/api/gpu-demo/stop", response_model=GpuDemoStatus)
def stop_gpu_demo() -> GpuDemoStatus:
    return GPU_DEMO.stop()


@app.post("/api/analyses", response_model=JobStatus, status_code=202)
def create_analysis(
    athlete_video: UploadFile = File(...),
    reference_video: Optional[UploadFile] = File(default=None),
    handedness: str = Form(default="right"),
) -> JobStatus:
    if handedness not in {"left", "right"}:
        raise HTTPException(status_code=400, detail="Handedness must be left or right.")

    analysis_id = str(uuid.uuid4())
    upload_directory = UPLOAD_DIR / analysis_id
    upload_directory.mkdir(parents=True, exist_ok=False)
    try:
        athlete_suffix = Path(athlete_video.filename or "").suffix.lower()
        athlete_path = upload_directory / f"athlete{athlete_suffix}"
        _save_upload(athlete_video, athlete_path)

        reference_path = None
        reference_filename = None
        if reference_video and reference_video.filename:
            reference_suffix = Path(reference_video.filename).suffix.lower()
            reference_path = upload_directory / f"reference{reference_suffix}"
            _save_upload(reference_video, reference_path)
            reference_filename = reference_video.filename
    except Exception:
        shutil.rmtree(upload_directory, ignore_errors=True)
        raise

    job = JobStatus(
        analysis_id=analysis_id,
        status="queued",
        progress=0,
        message="Analysis queued",
    )
    with JOBS_LOCK:
        JOBS[analysis_id] = job

    worker = threading.Thread(
        target=_process_job,
        args=(
            analysis_id,
            athlete_path,
            athlete_video.filename or athlete_path.name,
            handedness,
            reference_path,
            reference_filename,
        ),
        daemon=True,
    )
    worker.start()
    return job


@app.get("/api/analyses/{analysis_id}", response_model=JobStatus)
def get_analysis(analysis_id: str) -> JobStatus:
    with JOBS_LOCK:
        job = JOBS.get(analysis_id)
    if not job:
        result_path = RESULTS_DIR / f"{analysis_id}.json"
        if result_path.exists():
            return JobStatus(
                analysis_id=analysis_id,
                status="completed",
                progress=100,
                message="Analysis complete",
                result_url=f"/api/analyses/{analysis_id}/result",
            )
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return job


@app.get("/api/analyses/{analysis_id}/result", response_model=AnalysisResult)
def get_result(analysis_id: str) -> AnalysisResult:
    result_path = RESULTS_DIR / f"{analysis_id}.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Analysis result is not available.")
    result = AnalysisResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    if result.athlete.display_rotation_degrees is None:
        video_path = next((UPLOAD_DIR / analysis_id).glob("athlete.*"), None)
        if video_path is None:
            LOGGER.warning("Source video is missing; view orientation is unknown for %s", analysis_id)
        else:
            try:
                result.athlete.display_rotation_degrees = read_video_display_rotation(video_path)
            except PoseEstimationError:
                LOGGER.warning("View orientation is unavailable for %s", analysis_id, exc_info=True)
    return result


@app.get("/api/analyses/{analysis_id}/video")
def get_athlete_video(analysis_id: str) -> FileResponse:
    upload_directory = UPLOAD_DIR / analysis_id
    matches = list(upload_directory.glob("athlete.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Video not found.")
    return FileResponse(matches[0])
