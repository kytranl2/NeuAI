"""Entry point — orchestrates the inference pipeline."""

import signal
import sys
from pathlib import Path

import yaml

from camera import Camera
from engine import Engine
from monitor import Monitor
from postprocessor import Postprocessor
from preprocessor import Preprocessor


def load_config(path: str = "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def run(config: dict):
    model_cfg = config["model"]
    cam_cfg = config["camera"]
    inf_cfg = config["inference"]
    mon_cfg = config["monitor"]

    # Initialize components
    engine = Engine(
        model_path=model_cfg["path"],
        num_threads=inf_cfg["num_threads"],
        use_xnnpack=inf_cfg["use_xnnpack"],
    )
    preprocessor = Preprocessor(
        input_width=engine.input_width,
        input_height=engine.input_height,
        input_dtype=engine.input_dtype,
    )
    postprocessor = Postprocessor(
        labels_path=model_cfg["labels"],
        top_k=inf_cfg["top_k"],
    )
    monitor = Monitor(log_interval=mon_cfg["log_interval"])
    camera = Camera(
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        framerate=cam_cfg["framerate"],
    )

    # Graceful shutdown
    running = True

    def handle_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Starting inference pipeline (model: {model_cfg['path']})")
    print(f"Resolution: {cam_cfg['width']}x{cam_cfg['height']} | Threads: {inf_cfg['num_threads']}")
    print("Press Ctrl+C to stop.\n")

    camera.start()
    try:
        while running:
            frame = camera.capture()

            monitor.tick_start()
            input_tensor = preprocessor.process(frame)
            output = engine.predict(input_tensor)
            results = postprocessor.process(output)
            should_log = monitor.tick_end()

            # Print top prediction
            if results:
                label, confidence = results[0]
                print(f"  {label}: {confidence:.1%}", end="")
                if should_log:
                    print(f"  | {monitor.format_stats()}")
                else:
                    print()
    finally:
        camera.stop()
        print("\nShutdown complete.")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)
    run(config)


if __name__ == "__main__":
    main()
