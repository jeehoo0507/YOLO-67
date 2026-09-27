from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import psutil


def effective_cpus() -> int:
    count = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    try:
        count = min(count, len(psutil.Process().cpu_affinity()))
    except (AttributeError, psutil.Error):
        pass
    # Respect Linux cgroup v2 quotas (e.g. container with many visible host cores).
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, math.floor(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, count)


def available_ram() -> int:
    available = psutil.virtual_memory().available
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        used = int(Path("/sys/fs/cgroup/memory.current").read_text())
        if limit != "max":
            available = min(available, max(0, int(limit) - used))
    except (OSError, ValueError):
        pass
    return available


def worker_limit(max_workers: int = 0) -> int:
    physical = psutil.cpu_count(logical=False) or effective_cpus()
    # Reserve one effective core for decode/producer; allow 1 worker on a 1-core machine.
    cpu_limit = max(1, min(physical, effective_cpus() - 1))
    # NumPy/SciPy workers + 576^2 dense matrices; conservative 512 MiB per worker.
    memory_limit = max(1, int(available_ram() * 0.6) // (512 * 1024**2))
    return min(cpu_limit, memory_limit, max_workers or cpu_limit)


def worker_candidates(requested: list[int], limit: int) -> list[int]:
    if requested:
        return sorted({x for x in requested if x <= limit})
    counts = []
    value = 1
    while value <= limit:
        counts.append(value)
        value *= 2
    return sorted(set(counts + [limit]))


def system_info() -> dict:
    import numpy
    import scipy
    import torch

    info = {
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "effective_cpus": effective_cpus(),
        "ram_total_bytes": psutil.virtual_memory().total,
        "ram_available_bytes": available_ram(),
        "python": sys.version,
        "python_executable": sys.executable,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "gpus": [],
        "thread_env": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    for index in range(torch.cuda.device_count()):
        gpu = torch.cuda.get_device_properties(index)
        info["gpus"].append(
            {
                "index": index,
                "name": gpu.name,
                "vram_bytes": gpu.total_memory,
                "compute_capability": [gpu.major, gpu.minor],
            }
        )
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        info["nvidia_smi"] = result.stdout.strip() or result.stderr.strip()
    except (OSError, subprocess.TimeoutExpired):
        info["nvidia_smi"] = "unavailable"
    return info
