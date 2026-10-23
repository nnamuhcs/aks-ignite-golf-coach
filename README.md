# AKS Ignite 3D Golf Coach

This is a separate Ignite demo project. It does not modify `aks-edge-golf-ai`.

The application turns a real uploaded swing video into an interactive 3D coaching session. MediaPipe BlazePose GHUM produces image landmarks and metric scale world landmarks. The backend detects eight swing phases, calculates biomechanical measurements, reconstructs the wrist path, evaluates movement quality, and creates coaching guidance from the measured results. The Ignite interface does not use or display professional golfer footage. It renders the pose with Three.js and reads live AKS Flex Node, Kubernetes, host, and NVIDIA GPU status.

## Current experience

1. Upload an athlete swing.
2. Extract real 2D and 3D landmarks from sampled video frames.
3. Detect Address, Takeaway, Backswing, Top, Downswing, Impact, Follow Through, and Finish.
4. Measure shoulder turn, hip turn, spine tilt, lead arm extension, knee flex, tempo, wrist trajectory, and movement quality.
5. Review the source overlay and rotatable 3D reconstruction for every phase.
6. Read coaching guidance with the exact measurement used as evidence.
7. View the actual host, AKS node, Flex Node, and NVIDIA GPU status.

The 3D athlete, ghost poses, and wrist path retain the source video's camera-relative
orientation. Drag to rotate the model at any time. Each video Play event, including
resuming after a pause, resets the manual rotation to that source view without
changing the existing playback timing or seek synchronization.
Video display rotation is included in the analysis metadata, so portrait MOV
recordings also match the orientation shown by the browser. Existing saved
analyses recover that metadata from the original upload without rerunning inference.
If the source rotation cannot be recovered, the viewer keeps the legacy orientation
and explicitly reports that the video angle is unavailable.
Click the Golf Swing AI Coach logo or title to return to the main upload page.

No professional golfer footage or likeness is used by the Ignite experience. No platform value is simulated. If the application is not running in Kubernetes or cannot access an NVIDIA GPU, the interface says so.

## Local development

Requirements: Python 3.11 recommended, Python 3.9 minimum, Node.js 22 or newer.

```bash
make setup
make backend
make frontend
```

Open `http://localhost:5173`. Vite must proxy the API during local development, or set `VITE_API_BASE=http://localhost:8000`. The production frontend container proxies `/api` and `/assets` to the backend service.

Run validation:

```bash
make test
make lint
make build
```

Run both containers:

```bash
docker compose up --build
```

Open `http://localhost:8080`.

## Jetson Thor on AKS Flex Node

The Thor overlay pins the backend to `caldova-gpu-la`, requests one `nvidia.com/gpu` resource, and selects the `nvidia` runtime class:

```bash
kubectl apply -k deploy/overlays/jetson-thor
```

The cluster must provide the NVIDIA device plugin and an `nvidia` RuntimeClass before applying the overlay. Confirm these prerequisites:

```bash
kubectl get runtimeclass nvidia
kubectl get node caldova-gpu-la -o jsonpath='{.status.allocatable.nvidia\.com/gpu}'
```

GPU registration alone does not prove CUDA execution works. On Thor, the Flex Node worker and its workload CDI configuration must also expose the Tegra CUDA platform libraries, `/dev/nvmap`, and `/dev/nvsciipc`, with appropriate device permissions. The GPU image includes CUDA 13.0 development headers because CuPy compiles kernels at runtime. Validate actual video processing with `GPU_VIDEO_PROCESSING=required`, not just GPU enumeration.

Live GPU telemetry also requires the host's `nvidia-smi` executable to be available in the workload. Thor can report `[N/A]` for dedicated GPU memory; the dashboard preserves these metrics as unreported instead of displaying fabricated zero values or failing the platform endpoint.

The host kernel must support WireGuard and GENEVE for the Flex Node network used by this deployment. A Ready node and a locally healthy backend do not prove that an Azure-hosted frontend can reach Thor. Verify cross-node HTTP connectivity before deploying through Fleet.

The initial pose provider uses MediaPipe CPU inference. The Jetson image also runs real CUDA temporal motion and edge analysis in parallel through CuPy. That GPU output contributes its motion peak to impact timing and is reported separately from pose inference. The pose provider remains isolated in `backend/app/pose_estimator.py` so a TensorRT implementation can replace it without changing the API or frontend. Do not claim GPU accelerated pose inference until that provider is installed and the execution metadata reports it.

## AKS Fleet Manager deployment

The GitHub Actions workflow publishes these images to GHCR after every push to `main`:

```text
ghcr.io/nnamuhcs/aks-ignite-golf-coach-backend:latest
ghcr.io/nnamuhcs/aks-ignite-golf-coach-backend-gpu:latest
ghcr.io/nnamuhcs/aks-ignite-golf-coach-frontend:latest
```

For a portal-driven deployment without `kubectl`, open the `caldova-pharma-prod-aks-fleet` Fleet Manager resource and create an Automated Deployment. Use the existing `frontend/Dockerfile`, the `frontend` build context, ACR image `aks-ignite-golf-coach-frontend`, and the `fleet` manifest folder. The manifests place the complete application directly on member cluster `caldova-miami-cluster`; the backend remains pinned to GPU node `caldova-gpu-la`.

For a command-line deployment, connect to the Fleet hub cluster, stage the workload, and create the placement:

```bash
kubectl apply -k deploy/overlays/jetson-thor
kubectl apply -f deploy/fleet/cluster-resource-placement.yaml
```

Confirm placement and workload health:

```bash
kubectl get clusterresourceplacement aks-ignite-golf-coach
kubectl describe clusterresourceplacement aks-ignite-golf-coach
kubectl --context <gpu-member-context> get pods,services -n aks-ignite-golf-coach -o wide
```

The Fleet placement selects `caldova-miami-cluster`. Inside that cluster, the backend still schedules only on node `caldova-gpu-la` and requires the `nvidia` RuntimeClass plus one allocatable `nvidia.com/gpu`.

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/analyses` | Upload athlete and optional reference videos |
| `GET /api/analyses/{id}` | Read job status |
| `GET /api/analyses/{id}/result` | Read measured analysis |
| `GET /api/platform` | Read live host, GPU, and Kubernetes status |
| `GET /api/health` | Read service health |
