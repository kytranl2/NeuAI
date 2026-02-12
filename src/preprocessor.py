"""Image preprocessing for TFLite inference."""

import numpy as np


class Preprocessor:
    """Prepares camera frames for model input."""

    def __init__(self, input_width: int, input_height: int, input_dtype: np.dtype):
        self.input_width = input_width
        self.input_height = input_height
        self.input_dtype = input_dtype

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a frame for model input.

        For quantized uint8 models, raw pixel values are used directly.
        Returns array with batch dimension: (1, H, W, 3).
        """
        h, w = frame.shape[:2]

        if h != self.input_height or w != self.input_width:
            frame = _resize_nearest(frame, self.input_height, self.input_width)

        if frame.dtype != self.input_dtype:
            frame = frame.astype(self.input_dtype)

        # Add batch dimension using a view (zero-copy)
        return np.expand_dims(frame, axis=0)


def _resize_nearest(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize using nearest-neighbor interpolation (numpy only, no PIL/cv2)."""
    src_h, src_w = image.shape[:2]
    row_indices = (np.arange(target_h) * src_h // target_h).astype(int)
    col_indices = (np.arange(target_w) * src_w // target_w).astype(int)
    return image[np.ix_(row_indices, col_indices)]
