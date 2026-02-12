"""Tests for engine module."""

import importlib
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _create_mock_tflite_module():
    """Create a mock tflite_runtime module hierarchy."""
    mock_interpreter_cls = MagicMock()
    mock_interpreter = MagicMock()
    mock_interpreter_cls.return_value = mock_interpreter

    mock_interpreter.get_input_details.return_value = [{
        "shape": np.array([1, 224, 224, 3]),
        "dtype": np.uint8,
        "index": 0,
    }]
    mock_interpreter.get_output_details.return_value = [{
        "shape": np.array([1, 1001]),
        "dtype": np.uint8,
        "index": 1,
    }]

    input_buf = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    output_buf = np.random.randint(0, 256, (1, 1001), dtype=np.uint8)
    mock_interpreter.tensor.side_effect = lambda idx: (lambda: input_buf) if idx == 0 else (lambda: output_buf)

    mock_load_delegate = MagicMock(side_effect=OSError("no delegate"))

    # Build fake module
    tflite_runtime = ModuleType("tflite_runtime")
    tflite_interpreter = ModuleType("tflite_runtime.interpreter")
    tflite_interpreter.Interpreter = mock_interpreter_cls
    tflite_interpreter.load_delegate = mock_load_delegate
    tflite_runtime.interpreter = tflite_interpreter

    return tflite_runtime, tflite_interpreter, mock_interpreter


class TestEngineInit:
    def test_raises_without_tflite_runtime(self):
        # Remove tflite_runtime from sys.modules to trigger ImportError path
        saved = {}
        for key in list(sys.modules):
            if key.startswith("tflite_runtime"):
                saved[key] = sys.modules.pop(key)
        if "engine" in sys.modules:
            del sys.modules["engine"]

        try:
            import engine as eng
            importlib.reload(eng)
            with pytest.raises(RuntimeError, match="tflite-runtime"):
                eng.Engine("fake_model.tflite")
        finally:
            sys.modules.update(saved)


class TestEnginePrediction:
    def _load_engine_with_mock(self):
        tflite_runtime, tflite_interpreter, mock_interp = _create_mock_tflite_module()

        # Inject fake modules
        saved = {}
        for key in list(sys.modules):
            if key.startswith("tflite_runtime"):
                saved[key] = sys.modules.pop(key)
        if "engine" in sys.modules:
            saved["engine"] = sys.modules.pop("engine")

        sys.modules["tflite_runtime"] = tflite_runtime
        sys.modules["tflite_runtime.interpreter"] = tflite_interpreter

        import engine as eng
        importlib.reload(eng)

        e = eng.Engine("fake.tflite", num_threads=2, use_xnnpack=True)
        return e, mock_interp, saved

    def _cleanup(self, saved):
        for key in list(sys.modules):
            if key.startswith("tflite_runtime"):
                del sys.modules[key]
        if "engine" in sys.modules:
            del sys.modules["engine"]
        sys.modules.update(saved)

    def test_predict_returns_correct_shape(self):
        engine, _, saved = self._load_engine_with_mock()
        try:
            dummy = np.zeros((1, 224, 224, 3), dtype=np.uint8)
            result = engine.predict(dummy)
            assert result.shape == (1, 1001)
        finally:
            self._cleanup(saved)

    def test_predict_calls_invoke(self):
        engine, mock_interp, saved = self._load_engine_with_mock()
        try:
            dummy = np.zeros((1, 224, 224, 3), dtype=np.uint8)
            engine.predict(dummy)
            mock_interp.invoke.assert_called_once()
        finally:
            self._cleanup(saved)

    def test_input_properties(self):
        engine, _, saved = self._load_engine_with_mock()
        try:
            assert engine.input_height == 224
            assert engine.input_width == 224
        finally:
            self._cleanup(saved)
