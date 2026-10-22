"""MediaPipe pose extraction with real image and world landmarks."""

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp

from .config import MAX_ANALYSIS_FRAMES, MIN_POSE_FRAMES
from .schemas import Landmark, PoseFrame, VideoMetadata

LOGGER = logging.getLogger(__name__)


class PoseEstimationError(RuntimeError):
    """Raised when a usable pose sequence cannot be extracted."""


def _display_rotation(capture: cv2.VideoCapture) -> int:
    if capture.get(cv2.CAP_PROP_ORIENTATION_AUTO):
        return 0
    return int(capture.get(cv2.CAP_PROP_ORIENTATION_META)) % 360


def read_video_display_rotation(video_path: Path) -> int:
    """Read the display rotation still needed after OpenCV decoding."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise PoseEstimationError("The source video's display rotation could not be read.")
        return _display_rotation(capture)
    finally:
        capture.release()


class MediaPipePoseEstimator:
    """Extract a temporally sampled 3D pose sequence from a video."""

    backend_name = "MediaPipe BlazePose GHUM 3D"
    processing_device = "CPU"

    def analyze(self, video_path: Path, original_filename: str) -> Tuple[VideoMetadata, List[PoseFrame]]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise PoseEstimationError("OpenCV could not open the uploaded video.")

        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frame_count <= 0 or fps <= 0:
                raise PoseEstimationError("The uploaded video does not contain valid timing metadata.")

            sample_stride = max(1, frame_count // MAX_ANALYSIS_FRAMES)
            analyzed_frames = 0
            pose_frames: List[PoseFrame] = []

            with mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55,
            ) as pose:
                frame_index = 0
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    if frame_index % sample_stride != 0:
                        frame_index += 1
                        continue

                    analyzed_frames += 1
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = pose.process(rgb_frame)
                    if result.pose_landmarks and result.pose_world_landmarks:
                        pose_frames.append(
                            PoseFrame(
                                frame_index=frame_index,
                                timestamp_seconds=frame_index / fps,
                                image_landmarks=[
                                    Landmark(
                                        x=point.x,
                                        y=point.y,
                                        z=point.z,
                                        visibility=point.visibility,
                                    )
                                    for point in result.pose_landmarks.landmark
                                ],
                                world_landmarks=[
                                    Landmark(
                                        x=point.x,
                                        y=point.y,
                                        z=point.z,
                                        visibility=point.visibility,
                                    )
                                    for point in result.pose_world_landmarks.landmark
                                ],
                            )
                        )
                    frame_index += 1

            if len(pose_frames) < MIN_POSE_FRAMES:
                raise PoseEstimationError(
                    "A stable golfer pose was not detected in enough frames. "
                    "Use a fixed camera, keep the full body visible, and upload a complete swing."
                )

            metadata = VideoMetadata(
                filename=original_filename,
                duration_seconds=frame_count / fps,
                source_fps=fps,
                width=width,
                height=height,
                analyzed_frames=analyzed_frames,
                pose_frames=len(pose_frames),
                display_rotation_degrees=_display_rotation(capture),
            )
            LOGGER.info("Extracted %d pose frames from %s", len(pose_frames), original_filename)
            return metadata, pose_frames
        finally:
            capture.release()
