"""Render measured pose landmarks onto source frames."""

from pathlib import Path
from typing import Dict, List

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

from .schemas import PoseFrame


def render_phase_images(
    video_path: Path,
    frames: List[PoseFrame],
    phase_indices: Dict[str, int],
    output_directory: Path,
    analysis_id: str,
) -> Dict[str, str]:
    output_directory.mkdir(parents=True, exist_ok=True)
    requested = {frames[index].frame_index: phase for phase, index in phase_indices.items()}
    urls: Dict[str, str] = {}
    capture = cv2.VideoCapture(str(video_path))

    try:
        frame_index = 0
        while requested:
            success, image = capture.read()
            if not success:
                break
            phase = requested.pop(frame_index, None)
            if phase is None:
                frame_index += 1
                continue

            pose_frame = frames[phase_indices[phase]]
            landmark_list = landmark_pb2.NormalizedLandmarkList()
            for point in pose_frame.image_landmarks:
                landmark = landmark_list.landmark.add()
                landmark.x = point.x
                landmark.y = point.y
                landmark.z = point.z
                landmark.visibility = point.visibility

            mp.solutions.drawing_utils.draw_landmarks(
                image,
                landmark_list,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
            )
            cv2.putText(
                image,
                phase.replace("_", " ").title(),
                (28, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            filename = f"{phase}.jpg"
            cv2.imwrite(str(output_directory / filename), image)
            urls[phase] = f"/assets/{analysis_id}/{filename}"
            frame_index += 1
    finally:
        capture.release()

    missing = set(phase_indices) - set(urls)
    if missing:
        raise RuntimeError(f"Could not render phase images: {', '.join(sorted(missing))}")
    return urls
