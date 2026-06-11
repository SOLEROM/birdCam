"""RuntimeState: owns configs, pipeline, frame source, worker thread.

Pipelines and camera models are immutable; config updates build new objects
and swap references atomically under a lock. The render worker reads the
references once per iteration, so a swap mid-frame is harmless.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from bev.camera_model import CameraModel
from bev.debug_draw import draw_bev_grid, draw_coverage, draw_seams, draw_valid_mask
from bev.stitcher import BevPipeline
from configs_schema import (
    BevConfig,
    CamerasConfig,
    ConfigError,
    SceneConfig,
    load_yaml_config,
)
from frame_sources.base import FrameSource
from web.hub import FrameHub

log = logging.getLogger("bev.runtime")

OVERLAY_NAMES = ("grid", "coverage", "seams", "mask")


@dataclass
class AppSettings:
    config_dir: Path = Path("config")
    source: str = "synthetic"  # synthetic | webots | folder
    folder: Path | None = None
    fps: float = 12.0
    jpeg_quality: int = 80


def _models(cfg: CamerasConfig) -> dict[str, CameraModel]:
    return {n: CameraModel.from_config(n, c) for n, c in cfg.cameras.items()}


class RuntimeState:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        cfg_dir = Path(settings.config_dir)
        self.cameras_cfg = load_yaml_config(cfg_dir / "cameras.yaml", CamerasConfig)
        self.bev_cfg = load_yaml_config(cfg_dir / "bev.yaml", BevConfig)
        self.scene_cfg = load_yaml_config(cfg_dir / "scene.yaml", SceneConfig)

        self._swap_lock = threading.Lock()
        self.models = _models(self.cameras_cfg)
        self.pipeline = BevPipeline(
            self.models, self.bev_cfg, {n: c.yaw_deg for n, c in self.cameras_cfg.cameras.items()}
        )
        self.hub = FrameHub()
        self.overlays: set[str] = set()
        self.source: FrameSource | None = None

        self.stats = {"fps": 0.0, "render_ms": 0.0, "frame_age_s": 0.0, "seq": 0}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ----- lifecycle ------------------------------------------------------

    def attach_source(self, source: FrameSource) -> None:
        self.source = source

    def start(self) -> None:
        if self.source is None:
            raise RuntimeError("no frame source attached")
        self._thread = threading.Thread(target=self._worker, name="bev-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.source is not None:
            self.source.close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ----- config updates (atomic swaps) ----------------------------------

    def apply_cameras(self, cfg: CamerasConfig) -> None:
        models = _models(cfg)
        pipeline = BevPipeline(
            models, self.bev_cfg, {n: c.yaw_deg for n, c in cfg.cameras.items()}
        )
        with self._swap_lock:
            self.cameras_cfg = cfg
            self.models = models
            self.pipeline = pipeline
        if self.source is not None:
            self.source.update_cameras(models)

    def apply_bev(self, cfg: BevConfig) -> None:
        pipeline = BevPipeline(
            self.models, cfg, {n: c.yaw_deg for n, c in self.cameras_cfg.cameras.items()}
        )
        with self._swap_lock:
            self.bev_cfg = cfg
            self.pipeline = pipeline

    def apply_scene(self, cfg: SceneConfig) -> None:
        with self._swap_lock:
            self.scene_cfg = cfg
        update = getattr(self.source, "update_scene", None)
        if update is not None:
            update(cfg)

    def set_overlays(self, names: set[str]) -> None:
        unknown = names - set(OVERLAY_NAMES)
        if unknown:
            raise ConfigError(f"unknown overlays: {sorted(unknown)}")
        with self._swap_lock:
            self.overlays = set(names)

    def save_configs(self) -> dict[str, str]:
        cfg_dir = Path(self.settings.config_dir)
        out = {}
        for fname, model in (
            ("cameras.yaml", self.cameras_cfg),
            ("bev.yaml", self.bev_cfg),
            ("scene.yaml", self.scene_cfg),
        ):
            path = cfg_dir / fname
            path.write_text(yaml.safe_dump(model.model_dump(mode="json"), sort_keys=False))
            out[fname] = str(path)
        return out

    # ----- render worker ---------------------------------------------------

    def _encode(self, img: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(
            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return buf.tobytes()

    def _apply_overlays(self, img: np.ndarray, coverage: np.ndarray) -> np.ndarray:
        pipeline, bev_cfg, overlays = self.pipeline, self.bev_cfg, self.overlays
        if "coverage" in overlays:
            img = draw_coverage(img, pipeline.weights)
        if "mask" in overlays:
            img = draw_valid_mask(img, coverage)
        if "seams" in overlays:
            img = draw_seams(img, pipeline.weights)
        if "grid" in overlays:
            img = draw_bev_grid(img, bev_cfg)
        return img

    def _worker(self) -> None:
        ema_dt = None
        last_t = time.monotonic()
        while not self._stop.is_set():
            try:
                bundle = self.source.capture()
                if bundle is None:
                    if self._stop.is_set():
                        break
                    time.sleep(0.05)
                    continue
                pipeline = self.pipeline  # read once: consistent for this frame
                t0 = time.perf_counter()
                result = pipeline.render(bundle.frames)
                render_ms = (time.perf_counter() - t0) * 1000.0

                for name, frame in bundle.frames.items():
                    if frame is not None:
                        self.hub.publish(name, bundle.seq, self._encode(frame))
                bev_img = self._apply_overlays(result.image, result.coverage)
                self.hub.publish("bev", bundle.seq, self._encode(bev_img))

                now = time.monotonic()
                dt = now - last_t
                last_t = now
                ema_dt = dt if ema_dt is None else 0.9 * ema_dt + 0.1 * dt
                self.stats = {
                    "fps": round(1.0 / ema_dt, 1) if ema_dt and ema_dt > 0 else 0.0,
                    "render_ms": round(render_ms, 1),
                    "frame_age_s": round(time.time() - bundle.timestamp, 3),
                    "seq": bundle.seq,
                }
            except Exception:
                log.exception("render worker iteration failed")
                time.sleep(0.2)
