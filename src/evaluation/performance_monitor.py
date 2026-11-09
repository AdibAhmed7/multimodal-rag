import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
import psutil

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def get_gpu_memory() -> Optional[Dict[str, float]]:
    """Measure GPU VRAM consumption if CUDA is available."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return None
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 2),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
    }


class PerformanceMonitor:
    """Monitors execution latency, CPU/Process memory (RSS), and GPU usage."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.results: List[Dict[str, Any]] = []

    @contextmanager
    def measure(self, operation_name: str):
        start_time = time.perf_counter()
        gpu_before = get_gpu_memory()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            memory_mb = self.process.memory_info().rss / (1024 * 1024)
            gpu_after = get_gpu_memory()

            record: Dict[str, Any] = {
                "operation": operation_name,
                "duration_seconds": round(elapsed, 4),
                "process_memory_mb": round(memory_mb, 2),
            }
            if gpu_after:
                record["gpu_memory_mb"] = gpu_after
            self.results.append(record)

    def get_results(self) -> List[Dict[str, Any]]:
        return list(self.results)

    def clear(self) -> None:
        self.results.clear()
