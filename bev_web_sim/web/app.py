"""FastAPI application factory wiring source -> runtime -> streams -> UI."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from web.config_api import router as config_router
from web.runtime import AppSettings, RuntimeState
from web.streams import MEDIA_TYPE, STREAM_NAMES, mjpeg_generator

log = logging.getLogger("bev.app")
STATIC_DIR = Path(__file__).parent / "static"


def _build_source(state: RuntimeState):
    s = state.settings
    if s.source == "synthetic":
        from frame_sources.synthetic_source import SyntheticSource

        return SyntheticSource(state.models, state.scene_cfg, fps=s.fps)
    if s.source == "folder":
        from frame_sources.folder_source import FolderSource

        return FolderSource(s.folder, fps=s.fps)
    if s.source == "webots":
        from frame_sources.webots_source import WebotsSource

        return WebotsSource(state.models, fps=s.fps)
    raise ValueError(f"unknown source '{s.source}'")


def create_app(settings: AppSettings | None = None) -> FastAPI:
    settings = settings or AppSettings()
    state = RuntimeState(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.attach_source(_build_source(state))
        state.start()
        try:
            yield
        finally:
            state.stop()

    app = FastAPI(title="BEV Web Sim", lifespan=lifespan)
    app.state.runtime = state
    app.include_router(config_router)

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/stream/{name}")
    def stream(name: str, frames: int | None = None):
        if name not in STREAM_NAMES:
            raise HTTPException(status_code=404, detail=f"unknown stream '{name}'")
        if frames is not None and not (1 <= frames <= 10000):
            raise HTTPException(status_code=422, detail="frames must be in [1, 10000]")
        return StreamingResponse(
            mjpeg_generator(state.hub, name, max_frames=frames), media_type=MEDIA_TYPE
        )

    @app.get("/frame/{name}.jpg")
    def frame(name: str):
        if name not in STREAM_NAMES:
            raise HTTPException(status_code=404, detail=f"unknown stream '{name}'")
        item = state.hub.wait_for(name, -1, timeout=5.0)
        if item is None:
            raise HTTPException(status_code=503, detail="no frame available yet")
        return Response(content=item[1], media_type="image/jpeg")

    @app.websocket("/ws/status")
    async def ws_status(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                await ws.send_json(state.stats)
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            pass

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
