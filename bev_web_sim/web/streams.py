"""MJPEG streaming generator.

max_frames bounds the stream (used by tests — TestClient buffers entire
responses, so infinite streams would hang it — and useful with curl).
"""
from __future__ import annotations

import time
from typing import Iterator

from web.hub import FrameHub

BOUNDARY = "frame"
MEDIA_TYPE = f"multipart/x-mixed-replace; boundary={BOUNDARY}"
STREAM_NAMES = ("front", "rear", "left", "right", "bev")


def mjpeg_generator(
    hub: FrameHub, name: str, max_fps: float = 15.0, max_frames: int | None = None
) -> Iterator[bytes]:
    last_seq = -1
    min_interval = 1.0 / max_fps
    last_sent = 0.0
    sent = 0
    while max_frames is None or sent < max_frames:
        item = hub.wait_for(name, last_seq, timeout=1.0)
        if item is None:
            continue  # no new frame yet; keep the connection alive
        seq, jpeg = item
        last_seq = seq
        now = time.monotonic()
        if now - last_sent < min_interval:
            time.sleep(min_interval - (now - last_sent))
        last_sent = time.monotonic()
        sent += 1
        yield (
            b"--" + BOUNDARY.encode() + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
        )
