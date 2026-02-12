#!/usr/bin/env python3
"""Standalone benchmark: sweep thread counts and resolutions.

Uses synthetic input — no camera required.
"""

import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Add src to path for engine import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from engine import Engine


def benchmark_threads(model_path: str, num_runs: int = 100):
    """Benchmark inference across different thread counts."""
    print(f"Model: {model_path}")
    print(f"Runs per config: {num_runs}")
    print("-" * 50)

    results = []

    for threads in range(1, 5):
        engine = Engine(model_path=model_path, num_threads=threads, use_xnnpack=True)
        dummy_input = np.random.randint(
            0, 256, size=engine.input_shape, dtype=engine.input_dtype
        )

        # Warm up
        for _ in range(10):
            engine.predict(dummy_input)

        # Timed runs
        start = time.perf_counter()
        for _ in range(num_runs):
            engine.predict(dummy_input)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / num_runs) * 1000
        fps = num_runs / elapsed
        results.append((threads, avg_ms, fps))
        print(f"  threads={threads}: {avg_ms:.2f} ms/inference, {fps:.1f} FPS")

    print("-" * 50)
    best = min(results, key=lambda r: r[1])
    print(f"Optimal: {best[0]} threads ({best[1]:.2f} ms, {best[2]:.1f} FPS)")


def main():
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    model_path = str(Path(__file__).resolve().parent.parent / config["model"]["path"])

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Run scripts/download_model.sh first.")
        sys.exit(1)

    benchmark_threads(model_path)


if __name__ == "__main__":
    main()
