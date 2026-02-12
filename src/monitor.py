"""Performance metrics: FPS, latency, memory, CPU temperature."""

import time
from collections import deque
from pathlib import Path

import psutil


class Monitor:
    """Tracks and reports inference performance metrics."""

    def __init__(self, log_interval: int = 100, window_size: int = 100):
        self.log_interval = log_interval
        self._latencies = deque(maxlen=window_size)
        self._frame_count = 0
        self._start_time = None

    def tick_start(self):
        """Call before inference."""
        self._start_time = time.perf_counter()

    def tick_end(self):
        """Call after inference. Returns True if stats should be logged."""
        if self._start_time is None:
            return False
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
        self._latencies.append(elapsed_ms)
        self._frame_count += 1
        return self._frame_count % self.log_interval == 0

    def get_stats(self) -> dict:
        """Return current performance statistics."""
        if not self._latencies:
            return {}

        avg_latency = sum(self._latencies) / len(self._latencies)
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)

        return {
            "avg_latency_ms": round(avg_latency, 2),
            "fps": round(fps, 1),
            "memory_mb": round(memory_mb, 1),
            "cpu_temp_c": self._read_cpu_temp(),
            "frames": self._frame_count,
        }

    def format_stats(self) -> str:
        stats = self.get_stats()
        if not stats:
            return ""
        temp = stats["cpu_temp_c"]
        temp_str = f"{temp:.1f}C" if temp is not None else "N/A"
        return (
            f"[frame {stats['frames']}] "
            f"latency: {stats['avg_latency_ms']:.1f}ms | "
            f"FPS: {stats['fps']:.1f} | "
            f"mem: {stats['memory_mb']:.1f}MB | "
            f"temp: {temp_str}"
        )

    @staticmethod
    def _read_cpu_temp() -> float | None:
        """Read Raspberry Pi CPU temperature from sysfs."""
        thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")
        if thermal_path.exists():
            try:
                return int(thermal_path.read_text().strip()) / 1000.0
            except (ValueError, OSError):
                pass
        return None
