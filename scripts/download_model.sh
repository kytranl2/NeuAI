#!/usr/bin/env bash
# Download quantized MobileNetV2 TFLite model and ImageNet labels.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/../models"

MODEL_URL="https://storage.googleapis.com/tensorflow/tf-keras-applications/mobilenet_v2/mobilenet_v2_1.0_224_quantized.tflite"
LABELS_URL="https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"

MODEL_FILE="$MODEL_DIR/mobilenet_v2_1.0_224_quantized.tflite"
LABELS_FILE="$MODEL_DIR/labels.txt"

mkdir -p "$MODEL_DIR"

echo "Downloading MobileNetV2 quantized model..."
if command -v wget &>/dev/null; then
    wget -q --show-progress -O "$MODEL_FILE" "$MODEL_URL"
elif command -v curl &>/dev/null; then
    curl -L --progress-bar -o "$MODEL_FILE" "$MODEL_URL"
else
    echo "Error: wget or curl is required." >&2
    exit 1
fi

echo "Downloading ImageNet labels..."
if command -v wget &>/dev/null; then
    wget -q --show-progress -O "$LABELS_FILE" "$LABELS_URL"
else
    curl -L --progress-bar -o "$LABELS_FILE" "$LABELS_URL"
fi

echo ""
echo "Done! Files saved to $MODEL_DIR/"
ls -lh "$MODEL_FILE" "$LABELS_FILE"
