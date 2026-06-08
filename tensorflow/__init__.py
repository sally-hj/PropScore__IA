"""Project-local TensorFlow shim.

This repository does not require the real TensorFlow runtime. The system
TensorFlow build on some macOS machines can abort at import time because it was
compiled without the CPU features available on the host. Keeping this shim in
the project root ensures any accidental ``import tensorflow`` resolves here
instead of loading the crashing binary wheel.
"""

raise ImportError(
    "TensorFlow is intentionally disabled for PropScore_AI. "
    "The app uses PyTorch / OpenCV fallbacks instead."
)
