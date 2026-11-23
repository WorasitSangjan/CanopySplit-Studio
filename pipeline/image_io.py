# SPDX-License-Identifier: AGPL-3.0-only

"""Read and write OpenCV images using Python paths, including Unicode on Windows."""

from pathlib import Path

import cv2
import numpy as np


def read_image(path, flags=cv2.IMREAD_COLOR):
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags) if data else None


def write_image(path, image, params=()):
    path = Path(path)
    success, data = cv2.imencode(path.suffix.lower(), image, params)
    if not success:
        return False
    try:
        path.write_bytes(data.tobytes())
    except OSError:
        return False
    return True
