"""Tests for postprocessor module."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from postprocessor import Postprocessor


@pytest.fixture
def labels_file(tmp_path):
    labels = ["background", "cat", "dog", "car", "bird"]
    path = tmp_path / "labels.txt"
    path.write_text("\n".join(labels))
    return str(path)


@pytest.fixture
def numbered_labels_file(tmp_path):
    lines = ["0 background", "1 cat", "2 dog", "3 car", "4 bird"]
    path = tmp_path / "labels.txt"
    path.write_text("\n".join(lines))
    return str(path)


class TestPostprocessor:
    def test_top_k_count(self, labels_file):
        pp = Postprocessor(labels_path=labels_file, top_k=3)
        output = np.array([[10, 200, 150, 50, 100]], dtype=np.uint8)
        results = pp.process(output)
        assert len(results) == 3

    def test_top_k_exceeds_classes(self, labels_file):
        pp = Postprocessor(labels_path=labels_file, top_k=10)
        output = np.array([[10, 200, 150, 50, 100]], dtype=np.uint8)
        results = pp.process(output)
        assert len(results) == 5

    def test_sorted_descending(self, labels_file):
        pp = Postprocessor(labels_path=labels_file, top_k=5)
        output = np.array([[10, 200, 150, 50, 100]], dtype=np.uint8)
        results = pp.process(output)
        confidences = [c for _, c in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_correct_label_mapping(self, labels_file):
        pp = Postprocessor(labels_path=labels_file, top_k=1)
        output = np.array([[0, 0, 0, 255, 0]], dtype=np.uint8)
        results = pp.process(output)
        assert results[0][0] == "car"

    def test_numbered_label_format(self, numbered_labels_file):
        pp = Postprocessor(labels_path=numbered_labels_file, top_k=1)
        output = np.array([[0, 255, 0, 0, 0]], dtype=np.uint8)
        results = pp.process(output)
        assert results[0][0] == "cat"

    def test_confidence_normalized(self, labels_file):
        pp = Postprocessor(labels_path=labels_file, top_k=1)
        output = np.array([[0, 0, 0, 255, 0]], dtype=np.uint8)
        results = pp.process(output)
        assert 0.0 <= results[0][1] <= 1.0

    def test_flat_output_shape(self, labels_file):
        pp = Postprocessor(labels_path=labels_file, top_k=2)
        output = np.array([10, 200, 150, 50, 100], dtype=np.uint8)
        results = pp.process(output)
        assert len(results) == 2
        assert results[0][0] == "cat"
