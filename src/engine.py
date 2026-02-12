"""TFLite interpreter wrapper with XNNPACK delegate."""

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    from unittest.mock import MagicMock
    Interpreter = None
    load_delegate = None


class Engine:
    """Wraps TFLite interpreter with pre-allocated tensors and optional XNNPACK."""

    def __init__(self, model_path: str, num_threads: int = 4, use_xnnpack: bool = True):
        if Interpreter is None:
            raise RuntimeError(
                "tflite-runtime is not installed. "
                "Install with: pip install tflite-runtime"
            )

        delegates = []
        if use_xnnpack:
            try:
                delegates.append(load_delegate("libXNNPACK.so"))
            except (ValueError, OSError):
                # XNNPACK may be built-in or unavailable; continue without explicit delegate
                pass

        self._interpreter = Interpreter(
            model_path=model_path,
            num_threads=num_threads,
            experimental_delegates=delegates if delegates else None,
        )
        self._interpreter.allocate_tensors()

        input_detail = self._interpreter.get_input_details()[0]
        output_detail = self._interpreter.get_output_details()[0]

        self.input_shape = tuple(input_detail["shape"])
        self.input_dtype = input_detail["dtype"]
        self.output_shape = tuple(output_detail["shape"])
        self.output_dtype = output_detail["dtype"]

        self._input_index = input_detail["index"]
        self._output_index = output_detail["index"]

        # Pre-fetch tensor buffer references for zero-copy access
        self._input_tensor = self._interpreter.tensor(self._input_index)
        self._output_tensor = self._interpreter.tensor(self._output_index)

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        """Run inference on input_array and return raw output tensor.

        Uses zero-copy write to the input tensor buffer for minimal overhead.
        """
        # Write directly into the interpreter's input buffer
        np.copyto(self._input_tensor(), input_array)
        self._interpreter.invoke()
        # Return a copy of output to avoid buffer reuse issues
        return self._output_tensor().copy()

    @property
    def input_height(self) -> int:
        return self.input_shape[1]

    @property
    def input_width(self) -> int:
        return self.input_shape[2]
