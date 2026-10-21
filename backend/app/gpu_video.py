"""Optional CUDA video motion analysis that runs beside CPU pose inference."""

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .schemas import GpuVideoAnalysis

LOGGER = logging.getLogger(__name__)


class GpuVideoProcessor:
    """Compute real temporal motion and edge energy with CuPy when CUDA is available."""

    def process(self, video_path: Path) -> GpuVideoAnalysis:
        required = os.environ.get("GPU_VIDEO_PROCESSING", "auto").lower() == "required"

        def unavailable(detail: str) -> GpuVideoAnalysis:
            if required:
                raise RuntimeError(detail)
            return GpuVideoAnalysis(
                available=False,
                device="Unavailable",
                analyzed_frames=0,
                detail=detail,
            )

        try:
            import cupy as cp
        except ImportError:
            return unavailable("CuPy CUDA provider is not installed.")

        try:
            device_count = cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError as error:
            LOGGER.warning("CUDA runtime is unavailable: %s", error)
            return unavailable(f"CUDA runtime unavailable: {error}")
        if device_count < 1:
            return unavailable("No CUDA device was detected.")

        properties = cp.cuda.runtime.getDeviceProperties(0)
        raw_name = properties.get("name", "CUDA GPU")
        device_name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError("GPU video processor could not open the uploaded video.")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_index = 0
        previous_gray: Optional[object] = None
        motion_values = []
        peak_motion = -1.0
        peak_timestamp = 0.0

        try:
            with cp.cuda.Device(0):
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    resized = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                    gpu_frame = cp.asarray(resized, dtype=cp.float32)
                    gray = (
                        gpu_frame[:, :, 2] * 0.299
                        + gpu_frame[:, :, 1] * 0.587
                        + gpu_frame[:, :, 0] * 0.114
                    )

                    gradient_x = cp.diff(gray, axis=1)
                    gradient_y = cp.diff(gray, axis=0)
                    edge_energy = (
                        cp.mean(cp.abs(gradient_x))
                        + cp.mean(cp.abs(gradient_y))
                    ) / 2.0
                    temporal_energy = cp.float32(0.0)
                    if previous_gray is not None:
                        temporal_energy = cp.mean(cp.abs(gray - previous_gray))

                    motion_energy = float((temporal_energy + edge_energy * 0.2).item())
                    motion_values.append(motion_energy)
                    if motion_energy > peak_motion:
                        peak_motion = motion_energy
                        peak_timestamp = frame_index / fps if fps > 0 else 0.0

                    previous_gray = gray
                    frame_index += 1

                cp.cuda.Stream.null.synchronize()
        finally:
            capture.release()

        mean_motion = float(np.mean(motion_values)) if motion_values else 0.0
        return GpuVideoAnalysis(
            available=True,
            device=device_name,
            analyzed_frames=frame_index,
            peak_motion_timestamp=round(peak_timestamp, 3),
            mean_motion_energy=round(mean_motion, 4),
            detail="CUDA temporal motion and edge analysis completed.",
        )
