# NeuAI

Lightweight TensorFlow Lite inference pipeline for image classification on Raspberry Pi 4. Optimized for minimal latency and memory footprint under edge-device constraints.

## Features

- **Pi Camera integration** via picamera2 with direct capture at model resolution
- **XNNPACK-accelerated** TFLite inference on ARM CPU
- **Quantized uint8** models for ~4x smaller size and ~2-3x faster inference
- **Zero-copy input** buffer and pre-allocated tensors for minimal per-frame overhead
- **Real-time monitoring** — FPS, latency, memory usage, CPU temperature

## Project Structure

```
NeuAI/
├── config.yaml                # Runtime configuration
├── src/
│   ├── main.py                # Entry point — orchestrates the pipeline
│   ├── camera.py              # Pi Camera capture with picamera2
│   ├── preprocessor.py        # Image preprocessing (resize, normalize)
│   ├── engine.py              # TFLite interpreter wrapper with XNNPACK
│   ├── postprocessor.py       # Top-K predictions with labels
│   └── monitor.py             # Performance metrics (FPS, latency, memory, temp)
├── models/                    # Drop .tflite model + labels.txt here
├── scripts/
│   ├── download_model.sh      # Download MobileNetV2 quantized model + labels
│   └── benchmark.py           # Thread-count sweep benchmark
└── tests/
    ├── test_preprocessor.py
    ├── test_engine.py
    └── test_postprocessor.py
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download model

```bash
bash scripts/download_model.sh
```

### 3. Run inference

```bash
python src/main.py
```

Press `Ctrl+C` to stop.

### 4. Benchmark

Find the optimal thread count for your device (no camera required):

```bash
python scripts/benchmark.py
```

## Configuration

Edit `config.yaml` to adjust runtime parameters:

```yaml
model:
  path: models/mobilenet_v2_1.0_224_quantized.tflite
  labels: models/labels.txt
camera:
  width: 224
  height: 224
  framerate: 30
inference:
  num_threads: 4
  use_xnnpack: true
  top_k: 5
monitor:
  log_interval: 100
```

## Performance Optimizations

| Technique | Impact |
|---|---|
| Quantized uint8 model | ~4x smaller, ~2-3x faster than float32 |
| XNNPACK delegate | ~2x faster CPU inference on ARM |
| Capture at model resolution | Eliminates resize step |
| Pre-allocated tensors | Zero per-frame memory allocation |
| Zero-copy input buffer | Avoids numpy array copies |
| 4 threads (quad-core Pi 4) | Full CPU utilization |

## Tests

```bash
pytest tests/ -v
```

## Requirements

- Raspberry Pi 4 (recommended)
- Pi Camera Module
- Python 3.10+
