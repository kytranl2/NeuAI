"""Pi Camera capture with picamera2, frame buffering."""

import numpy as np

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


class Camera:
    """Captures frames from Pi Camera at the specified resolution."""

    def __init__(self, width: int, height: int, framerate: int):
        if Picamera2 is None:
            raise RuntimeError(
                "picamera2 is not available. "
                "Install it on a Raspberry Pi with: pip install picamera2"
            )
        self.width = width
        self.height = height
        self.framerate = framerate
        self._camera = Picamera2()
        config = self._camera.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": framerate},
        )
        self._camera.configure(config)

    def start(self):
        self._camera.start()

    def stop(self):
        self._camera.stop()
        self._camera.close()

    def capture(self) -> np.ndarray:
        """Capture a single frame as an RGB uint8 numpy array (H, W, 3)."""
        return self._camera.capture_array("main")

    def stream(self):
        """Generator that yields RGB frames continuously."""
        self.start()
        try:
            while True:
                yield self.capture()
        finally:
            self.stop()
