"""End-to-end golf swing analysis pipeline."""

import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .annotator import render_phase_images
from .coaching import build_coaching
from .config import ASSETS_DIR, PHASE_DISPLAY_NAMES
from .gpu_video import GpuVideoProcessor
from .metrics import (
    calculate_phase_metrics,
    calculate_tempo,
    calculate_trajectory,
    compare_metrics,
    reference_score,
)
from .phases import detect_phase_indices, phase_confidence
from .platform import get_platform_status
from .pose_estimator import MediaPipePoseEstimator
from .scoring import overall_biomechanics_score, phase_biomechanics_score
from .smoothing import smooth_world_landmarks
from .schemas import (
    AnalysisResult,
    ExecutionMetadata,
    PhaseResult,
    PoseFrame,
    PoseSequenceFrame,
    VideoMetadata,
)


def _analyze_video(
    estimator: MediaPipePoseEstimator,
    video_path: Path,
    original_filename: str,
    handedness: str,
) -> Tuple[VideoMetadata, List[PoseFrame], Dict[str, int], Dict[str, Dict[str, float]]]:
    metadata, frames = estimator.analyze(video_path, original_filename)
    frames = smooth_world_landmarks(frames)
    indices, metrics = _measure_pose_frames(frames, handedness)
    return metadata, frames, indices, metrics


def _measure_pose_frames(
    frames: List[PoseFrame],
    handedness: str,
    motion_peak_timestamp: Optional[float] = None,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, float]]]:
    indices = detect_phase_indices(frames, motion_peak_timestamp)
    metrics = {
        phase: calculate_phase_metrics(frames[index], handedness)
        for phase, index in indices.items()
    }
    for metric_name in ("shoulder_turn", "hip_turn"):
        baseline = metrics["address"][metric_name]
        for phase in metrics:
            difference = abs(metrics[phase][metric_name] - baseline)
            metrics[phase][metric_name] = round(min(difference, 180.0 - difference), 1)
    return indices, metrics


def run_analysis(
    analysis_id: str,
    athlete_path: Path,
    athlete_filename: str,
    handedness: str,
    reference_path: Optional[Path] = None,
    reference_filename: Optional[str] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> AnalysisResult:
    started = time.monotonic()
    estimator = MediaPipePoseEstimator()
    if progress_callback:
        progress_callback(15, "Sampling source video frames")
    gpu_processor = GpuVideoProcessor()
    with ThreadPoolExecutor(max_workers=1) as executor:
        gpu_future = executor.submit(gpu_processor.process, athlete_path)
        athlete_metadata, athlete_frames = estimator.analyze(athlete_path, athlete_filename)
        gpu_video_analysis = gpu_future.result()
    athlete_frames = smooth_world_landmarks(athlete_frames)
    phase_indices, phase_metrics = _measure_pose_frames(
        athlete_frames,
        handedness,
        gpu_video_analysis.peak_motion_timestamp,
    )
    if progress_callback:
        progress_callback(48, "3D body landmarks extracted")

    image_urls = render_phase_images(
        athlete_path,
        athlete_frames,
        phase_indices,
        ASSETS_DIR / analysis_id,
        analysis_id,
    )
    if progress_callback:
        progress_callback(64, "Eight swing phases detected")

    reference_metadata = None
    reference_metrics: Dict[str, Dict[str, float]] = {}
    if reference_path and reference_filename:
        reference_metadata, _, _, reference_metrics = _analyze_video(
            estimator,
            reference_path,
            reference_filename,
            handedness,
        )
    if progress_callback:
        progress_callback(78, "Calculating measured biomechanics")

    comparison = compare_metrics(phase_metrics, reference_metrics)
    tempo_ratio = calculate_tempo(
        athlete_frames,
        phase_indices["top"],
        phase_indices["impact"],
    )
    phases = [
        PhaseResult(
            name=phase,
            display_name=PHASE_DISPLAY_NAMES[phase],
            frame_index=athlete_frames[index].frame_index,
            timestamp_seconds=round(athlete_frames[index].timestamp_seconds, 3),
            confidence=round(phase_confidence(athlete_frames[index]), 3),
            score=phase_biomechanics_score(
                phase,
                phase_metrics[phase],
                phase_confidence(athlete_frames[index]),
            ),
            image_url=image_urls[phase],
            metrics=phase_metrics[phase],
            world_landmarks=athlete_frames[index].world_landmarks,
        )
        for phase, index in phase_indices.items()
    ]
    if progress_callback:
        progress_callback(90, "Building coaching and synchronized replay")
    platform_status = get_platform_status()

    return AnalysisResult(
        analysis_id=analysis_id,
        handedness=handedness,
        athlete=athlete_metadata,
        reference=reference_metadata,
        phases=phases,
        pose_sequence=[
            PoseSequenceFrame(
                timestamp_seconds=round(frame.timestamp_seconds, 3),
                world_landmarks=frame.world_landmarks,
            )
            for frame in athlete_frames
        ],
        trajectory=calculate_trajectory(athlete_frames),
        comparison=comparison,
        coaching=build_coaching(phase_metrics, tempo_ratio, comparison),
        tempo_ratio=tempo_ratio,
        overall_score=overall_biomechanics_score(phases, tempo_ratio),
        reference_score=reference_score(comparison),
        gpu_video_analysis=gpu_video_analysis,
        execution=ExecutionMetadata(
            pose_backend=estimator.backend_name,
            processing_device=estimator.processing_device,
            processing_seconds=round(time.monotonic() - started, 2),
            host_architecture=platform.machine(),
            gpu_name=platform_status.gpu.name,
            gpu_video_processing=gpu_video_analysis.available,
            video_processing_device=gpu_video_analysis.device,
        ),
    )
