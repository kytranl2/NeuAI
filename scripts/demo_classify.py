#!/usr/bin/env python3
"""Demo: classify images using the NeuAI pipeline (no camera required).

Downloads sample images and runs inference, printing top-5 predictions.
"""

import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Use full tensorflow's tflite interpreter if tflite-runtime isn't available
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# Patch engine module to use the correct Interpreter
import engine
engine.Interpreter = Interpreter
engine.load_delegate = lambda *a, **kw: (_ for _ in ()).throw(OSError("skip"))

import yaml
from preprocessor import Preprocessor
from postprocessor import Postprocessor
from engine import Engine


SAMPLE_IMAGES = {
    "cat": "https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg",
    "dog": "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg",
    "military_uniform": "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg",
}


def download_image(url: str, save_path: Path) -> Path:
    if not save_path.exists():
        print(f"  Downloading {save_path.name}...")
        req = urllib.request.Request(url, headers={"User-Agent": "NeuAI-Demo/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            save_path.write_bytes(resp.read())
    return save_path


def load_and_resize(path: Path, width: int, height: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((width, height), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def main():
    project_root = Path(__file__).resolve().parent.parent
    config = yaml.safe_load((project_root / "config.yaml").read_text())

    model_path = str(project_root / config["model"]["path"])
    labels_path = str(project_root / config["model"]["labels"])
    top_k = config["inference"]["top_k"]
    num_threads = config["inference"]["num_threads"]

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Run: bash scripts/download_model.sh")
        sys.exit(1)

    # Initialize pipeline components
    print("Loading model...")
    eng = Engine(model_path=model_path, num_threads=num_threads, use_xnnpack=False)
    preprocessor = Preprocessor(
        input_width=eng.input_width,
        input_height=eng.input_height,
        input_dtype=eng.input_dtype,
    )
    postprocessor = Postprocessor(labels_path=labels_path, top_k=top_k)

    print(f"Model input: {eng.input_shape} ({eng.input_dtype})")
    print(f"Model output: {eng.output_shape} ({eng.output_dtype})")
    print()

    # Download and classify sample images
    tmp_dir = project_root / "tmp_demo_images"
    tmp_dir.mkdir(exist_ok=True)

    for name, url in SAMPLE_IMAGES.items():
        print(f"{'=' * 50}")
        print(f"Image: {name}")
        print(f"{'=' * 50}")

        try:
            img_path = download_image(url, tmp_dir / f"{name}.jpg")
            frame = load_and_resize(img_path, eng.input_width, eng.input_height)

            # Run pipeline
            input_tensor = preprocessor.process(frame)

            start = time.perf_counter()
            output = eng.predict(input_tensor)
            latency_ms = (time.perf_counter() - start) * 1000

            results = postprocessor.process(output)

            # Display results
            print(f"  Inference time: {latency_ms:.1f} ms")
            print(f"  Top-{top_k} predictions:")
            for i, (label, confidence) in enumerate(results, 1):
                bar = "#" * int(confidence * 30)
                print(f"    {i}. {label:<30s} {confidence:6.2%}  {bar}")
            print()

        except Exception as e:
            print(f"  Error: {e}")
            print()

    # Cleanup
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()


if __name__ == "__main__":
    main()
