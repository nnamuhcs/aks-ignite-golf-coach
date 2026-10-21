"""Live host, NVIDIA GPU, and Kubernetes status discovery."""

import logging
import os
import platform
import socket
import subprocess
import psutil
from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException

from .schemas import (
    GpuStatus,
    KubernetesInventory,
    KubernetesService,
    KubernetesStatus,
    KubernetesWorkload,
    PlatformStatus,
)

LOGGER = logging.getLogger(__name__)


def _gpu_status() -> GpuStatus:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=4)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return GpuStatus(available=False)

    first_line = completed.stdout.strip().splitlines()[0]
    fields = [field.strip() for field in first_line.split(",")]
    if len(fields) != 5:
        return GpuStatus(available=False)
    return GpuStatus(
        available=True,
        name=fields[0],
        driver_version=fields[1],
        memory_total_mb=int(fields[2]),
        memory_used_mb=int(fields[3]),
        utilization_percent=int(fields[4]),
    )


def _kubernetes_status() -> KubernetesStatus:
    namespace = os.environ.get("POD_NAMESPACE")
    pod_name = os.environ.get("POD_NAME", socket.gethostname())
    node_name = os.environ.get("NODE_NAME")
    try:
        config.load_incluster_config()
    except ConfigException:
        return KubernetesStatus(in_cluster=False)

    try:
        api = client.CoreV1Api()
        if not node_name and namespace:
            pod = api.read_namespaced_pod(name=pod_name, namespace=namespace)
            node_name = pod.spec.node_name
        if not node_name:
            return KubernetesStatus(in_cluster=True, namespace=namespace, pod_name=pod_name)

        node = api.read_node(name=node_name)
        labels = node.metadata.labels or {}
        return KubernetesStatus(
            in_cluster=True,
            namespace=namespace,
            pod_name=pod_name,
            node_name=node_name,
            node_pool=labels.get("kubernetes.azure.com/agentpool"),
            node_architecture=labels.get("kubernetes.io/arch"),
            node_os=labels.get("kubernetes.io/os"),
            flex_node=(
                "net.unbounded-cloud.io/site" in labels
                or labels.get("kubernetes.azure.com/mode") == "flex"
            ),
        )
    except client.ApiException as error:
        LOGGER.warning("Kubernetes status lookup failed: %s", error)
        return KubernetesStatus(
            in_cluster=True,
            namespace=namespace,
            pod_name=pod_name,
            node_name=node_name,
        )


def get_platform_status() -> PlatformStatus:
    memory = psutil.virtual_memory()
    return PlatformStatus(
        hostname=socket.gethostname(),
        architecture=platform.machine(),
        cpu_count=psutil.cpu_count(logical=True) or 0,
        cpu_utilization_percent=round(psutil.cpu_percent(interval=0.1), 1),
        memory_total_mb=int(memory.total / (1024 * 1024)),
        memory_used_mb=int(memory.used / (1024 * 1024)),
        gpu=_gpu_status(),
        kubernetes=_kubernetes_status(),
    )


def get_kubernetes_inventory() -> KubernetesInventory:
    namespace = os.environ.get("POD_NAMESPACE")
    pod_name = os.environ.get("POD_NAME", socket.gethostname())
    try:
        config.load_incluster_config()
    except ConfigException:
        return KubernetesInventory(
            in_cluster=False,
            pods=[],
            deployments=[],
            services=[],
        )

    namespace = namespace or "default"
    core_api = client.CoreV1Api()
    apps_api = client.AppsV1Api()
    try:
        current_pod = core_api.read_namespaced_pod(name=pod_name, namespace=namespace)
        node_name = current_pod.spec.node_name
        node_pool = None
        if node_name:
            node = core_api.read_node(name=node_name)
            node_pool = (node.metadata.labels or {}).get("kubernetes.azure.com/agentpool")

        pods = []
        for pod in core_api.list_namespaced_pod(namespace=namespace).items:
            statuses = pod.status.container_statuses or []
            ready = sum(1 for status in statuses if status.ready)
            restarts = sum(status.restart_count for status in statuses)
            pods.append(
                KubernetesWorkload(
                    name=pod.metadata.name,
                    kind="Pod",
                    status=pod.status.phase or "Unknown",
                    ready=ready,
                    desired=len(pod.spec.containers or []),
                    restarts=restarts,
                    node_name=pod.spec.node_name,
                )
            )

        deployments = []
        for deployment in apps_api.list_namespaced_deployment(namespace=namespace).items:
            deployments.append(
                KubernetesWorkload(
                    name=deployment.metadata.name,
                    kind="Deployment",
                    status="Available" if (deployment.status.available_replicas or 0) > 0 else "Unavailable",
                    ready=deployment.status.ready_replicas or 0,
                    desired=deployment.spec.replicas or 0,
                )
            )

        services = []
        for service in core_api.list_namespaced_service(namespace=namespace).items:
            ports = [
                f"{port.port}/{port.protocol or 'TCP'}"
                for port in (service.spec.ports or [])
            ]
            services.append(
                KubernetesService(
                    name=service.metadata.name,
                    service_type=service.spec.type or "ClusterIP",
                    cluster_ip=service.spec.cluster_ip,
                    ports=ports,
                )
            )

        return KubernetesInventory(
            in_cluster=True,
            namespace=namespace,
            node_name=node_name,
            node_pool=node_pool,
            pods=pods,
            deployments=deployments,
            services=services,
        )
    except client.ApiException as error:
        LOGGER.warning("Kubernetes inventory lookup failed: %s", error)
        return KubernetesInventory(
            in_cluster=True,
            namespace=namespace,
            pods=[],
            deployments=[],
            services=[],
        )
