"""Decode predictions, top-K results, labels."""

from pathlib import Path

import numpy as np


class Postprocessor:
    """Decodes model output into labeled predictions."""

    def __init__(self, labels_path: str, top_k: int = 5):
        self.top_k = top_k
        self.labels = self._load_labels(labels_path)

    def process(self, output: np.ndarray) -> list[tuple[str, float]]:
        """Extract top-K predictions from model output.

        Args:
            output: Raw model output array, shape (1, num_classes) or (num_classes,).

        Returns:
            List of (label, confidence) tuples sorted by confidence descending.
        """
        scores = output.flatten().astype(np.float32)

        # For quantized uint8 output, normalize to [0, 1]
        if scores.max() > 1.0:
            scores = scores / 255.0

        k = min(self.top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
            results.append((label, float(scores[idx])))
        return results

    @staticmethod
    def _load_labels(path: str) -> list[str]:
        text = Path(path).read_text()
        labels = []
        for line in text.strip().splitlines():
            # Handle formats like "0 label" or just "label"
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1])
            else:
                labels.append(line.strip())
        return labels
