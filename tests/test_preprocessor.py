"""Tests for preprocessor module."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessor import Preprocessor, _resize_nearest


class TestPreprocessor:
    def test_output_shape_and_batch_dim(self):
        pp = Preprocessor(input_width=224, input_height=224, input_dtype=np.uint8)
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        result = pp.process(frame)
        assert result.shape == (1, 224, 224, 3)

    def test_dtype_preserved_uint8(self):
        pp = Preprocessor(input_width=224, input_height=224, input_dtype=np.uint8)
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        result = pp.process(frame)
        assert result.dtype == np.uint8

    def test_dtype_conversion_to_float32(self):
        pp = Preprocessor(input_width=224, input_height=224, input_dtype=np.float32)
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        result = pp.process(frame)
        assert result.dtype == np.float32

    def test_resize_when_dimensions_differ(self):
        pp = Preprocessor(input_width=224, input_height=224, input_dtype=np.uint8)
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = pp.process(frame)
        assert result.shape == (1, 224, 224, 3)

    def test_no_resize_when_dimensions_match(self):
        pp = Preprocessor(input_width=224, input_height=224, input_dtype=np.uint8)
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        result = pp.process(frame)
        # Should be a view of the same data (zero-copy)
        assert np.shares_memory(result, frame)


class TestResizeNearest:
    def test_downscale(self):
        img = np.arange(12).reshape(3, 4, 1).astype(np.uint8)
        resized = _resize_nearest(img, 2, 2)
        assert resized.shape == (2, 2, 1)

    def test_upscale(self):
        img = np.arange(3).reshape(1, 3, 1).astype(np.uint8)
        resized = _resize_nearest(img, 2, 6)
        assert resized.shape == (2, 6, 1)

    def test_preserves_dtype(self):
        img = np.zeros((10, 10, 3), dtype=np.float32)
        resized = _resize_nearest(img, 5, 5)
        assert resized.dtype == np.float32
