# NeuAI

Lightweight TensorFlow Lite inference pipeline for image classification on Raspberry Pi 4. Optimized for minimal latency and memory footprint under edge-device constraints.

## Features

- **Pi Camera integration** via picamera2 with direct capture at model resolution
- **XNNPACK-accelerated** TFLite inference on ARM CPU
- **Quantized uint8** models for ~4x smaller size and ~2-3x faster inference
- **Zero-copy input** buffer and pre-allocated tensors for minimal per-frame overhead
- **Real-time monitoring** — FPS, latency, memory usage, CPU temperature
- **Desktop demo** — classify images without a Pi using the included demo script

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
│   ├── benchmark.py           # Thread-count sweep benchmark
│   └── demo_classify.py       # Classify sample images (no camera required)
└── tests/
    ├── test_preprocessor.py
    ├── test_engine.py
    └── test_postprocessor.py
```

## Quick Start

### 1. Install dependencies

On Raspberry Pi:
```bash
pip install -r requirements.txt
```

On macOS/Linux desktop (for testing without a Pi):
```bash
pip install tensorflow numpy pyyaml psutil pillow
```

### 2. Download model

```bash
bash scripts/download_model.sh
```

This downloads a quantized MobileNetV2 model (~3.4 MB) and ImageNet labels (1001 classes).

### 3. Run inference

**On Raspberry Pi** (with Pi Camera):
```bash
python src/main.py
```

Press `Ctrl+C` to stop.

**On desktop** (no camera required):
```bash
python scripts/demo_classify.py
```

Downloads sample images and classifies them, showing top-5 predictions:

```
==================================================
Image: cat
==================================================
  Inference time: 5.1 ms
  Top-5 predictions:
    1. lynx                           76.47%  ######################
    2. Egyptian cat                   65.10%  ###################
    3. tiger cat                      58.82%  #################
    4. tabby                          57.25%  #################
    5. grey fox                       53.73%  ################

==================================================
Image: dog
==================================================
  Inference time: 5.1 ms
  Top-5 predictions:
    1. Labrador retriever             61.57%  ##################
    2. Saluki                         60.00%  ##################
    3. Ibizan hound                   57.65%  #################
    4. Eskimo dog                     53.73%  ################
    5. Siberian husky                 49.80%  ##############

==================================================
Image: military_uniform
==================================================
  Inference time: 3.9 ms
  Top-5 predictions:
    1. military uniform               70.20%  #####################
    2. mortarboard                    50.98%  ###############
    3. racket                         50.98%  ###############
    4. bearskin                       50.59%  ###############
    5. bow tie                        50.20%  ###############
```

### 4. Benchmark

Find the optimal thread count for your device (no camera required):

```bash
python scripts/benchmark.py
```

## Configuration

Edit `config.yaml` to adjust runtime parameters:

```yaml
model:
  path: models/mobilenet_v2_1.0_224_quant.tflite
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

All 19 tests run without a Pi or TFLite runtime (uses mocks):

```bash
pytest tests/ -v
```

```
tests/test_engine.py       — 4 tests (init, predict shape, invoke, properties)
tests/test_postprocessor.py — 7 tests (top-K, sorting, labels, normalization)
tests/test_preprocessor.py — 8 tests (shape, dtype, resize, zero-copy)
```

## Requirements

- Python 3.10+
- Raspberry Pi 4 + Pi Camera Module (for live inference)
- Or any desktop machine (for demo classification and tests)
