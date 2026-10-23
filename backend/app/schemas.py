"""API and analysis data models."""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Landmark(BaseModel):
    x: float
    y: float
    z: float
    visibility: float = 1.0


class PoseFrame(BaseModel):
    frame_index: int
    timestamp_seconds: float
    image_landmarks: List[Landmark]
    world_landmarks: List[Landmark]


class PhaseResult(BaseModel):
    name: str
    display_name: str
    frame_index: int
    timestamp_seconds: float
    confidence: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=100.0)
    image_url: str
    metrics: Dict[str, float]
    world_landmarks: List[Landmark]


class TrajectoryPoint(BaseModel):
    timestamp_seconds: float
    x: float
    y: float
    z: float


class PoseSequenceFrame(BaseModel):
    timestamp_seconds: float
    world_landmarks: List[Landmark]


class ComparisonMetric(BaseModel):
    metric: str
    athlete_value: float
    reference_value: float
    difference: float
    unit: str


class CoachingInsight(BaseModel):
    phase: str
    title: str
    detail: str
    severity: Literal["strength", "focus", "information"]
    evidence: str


class VideoMetadata(BaseModel):
    filename: str
    duration_seconds: float
    source_fps: float
    width: int
    height: int
    analyzed_frames: int
    pose_frames: int
    display_rotation_degrees: Optional[int] = Field(default=None, ge=0, lt=360)


class ExecutionMetadata(BaseModel):
    pose_backend: str
    processing_device: str
    processing_seconds: float
    host_architecture: str
    gpu_name: Optional[str] = None
    gpu_video_processing: bool = False
    video_processing_device: str = "Unavailable"


class GpuVideoAnalysis(BaseModel):
    available: bool
    device: str
    analyzed_frames: int
    peak_motion_timestamp: Optional[float] = None
    mean_motion_energy: Optional[float] = None
    detail: str


class AnalysisResult(BaseModel):
    analysis_id: str
    handedness: Literal["left", "right"]
    athlete: VideoMetadata
    reference: Optional[VideoMetadata] = None
    phases: List[PhaseResult]
    pose_sequence: List[PoseSequenceFrame]
    trajectory: List[TrajectoryPoint]
    comparison: List[ComparisonMetric]
    coaching: List[CoachingInsight]
    tempo_ratio: Optional[float] = None
    overall_score: float = Field(ge=0.0, le=100.0)
    reference_score: Optional[float] = None
    gpu_video_analysis: GpuVideoAnalysis
    execution: ExecutionMetadata


class JobStatus(BaseModel):
    analysis_id: str
    status: Literal["queued", "analyzing", "completed", "failed"]
    progress: int = Field(ge=0, le=100)
    message: str
    result_url: Optional[str] = None
    error: Optional[str] = None


class GpuStatus(BaseModel):
    available: bool
    name: Optional[str] = None
    driver_version: Optional[str] = None
    memory_total_mb: Optional[int] = None
    memory_used_mb: Optional[int] = None
    utilization_percent: Optional[int] = None


class KubernetesStatus(BaseModel):
    in_cluster: bool
    namespace: Optional[str] = None
    pod_name: Optional[str] = None
    node_name: Optional[str] = None
    node_pool: Optional[str] = None
    node_architecture: Optional[str] = None
    node_os: Optional[str] = None
    flex_node: Optional[bool] = None


class PlatformStatus(BaseModel):
    hostname: str
    architecture: str
    cpu_count: int
    cpu_utilization_percent: float
    memory_total_mb: int
    memory_used_mb: int
    gpu: GpuStatus
    kubernetes: KubernetesStatus


class KubernetesWorkload(BaseModel):
    name: str
    kind: Literal["Pod", "Deployment"]
    status: str
    ready: int
    desired: int
    restarts: int = 0
    node_name: Optional[str] = None


class KubernetesService(BaseModel):
    name: str
    service_type: str
    cluster_ip: Optional[str] = None
    ports: List[str]


class KubernetesInventory(BaseModel):
    in_cluster: bool
    namespace: Optional[str] = None
    node_name: Optional[str] = None
    node_pool: Optional[str] = None
    pods: List[KubernetesWorkload]
    deployments: List[KubernetesWorkload]
    services: List[KubernetesService]


class GpuDemoStartRequest(BaseModel):
    duration_seconds: int = Field(default=120, ge=5, le=120)


class GpuDemoStatus(BaseModel):
    """Manual CUDA demo load state. Not related to golf swing analysis results."""

    label: str
    state: Literal["idle", "starting", "running", "stopping", "stopped", "completed", "failed"]
    active: bool
    run_id: Optional[str] = None
    pid: Optional[int] = None
    device_name: Optional[str] = None
    matrix_size: Optional[int] = None
    allocated_memory_mb: Optional[float] = None
    memory_limit_mb: int
    duty_cycle: float
    max_duration_seconds: int
    requested_duration_seconds: Optional[int] = None
    started_at: Optional[datetime] = None
    elapsed_seconds: Optional[float] = None
    remaining_seconds: Optional[float] = None
    iterations: Optional[int] = None
    gpu_busy_seconds: Optional[float] = None
    measured_tflops: Optional[float] = None
    stop_reason: Optional[Literal["user", "time_limit", "watchdog", "shutdown", "error"]] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    message: str
