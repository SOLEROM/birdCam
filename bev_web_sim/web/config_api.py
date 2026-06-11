"""Config + debug REST endpoints (validation via the pydantic schemas)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from bev.debug_draw import coverage_footprints
from configs_schema import BevConfig, CamerasConfig, ConfigError, SceneConfig
from web.rate_limit import RateLimiter
from web.runtime import OVERLAY_NAMES, RuntimeState

router = APIRouter()
_limiter = RateLimiter(max_requests=30, window_s=1.0)


def _state(request: Request) -> RuntimeState:
    return request.app.state.runtime


@router.get("/config/cameras")
def get_cameras(request: Request):
    return _state(request).cameras_cfg.model_dump(mode="json")


@router.post("/config/cameras", dependencies=[Depends(_limiter)])
def post_cameras(cfg: CamerasConfig, request: Request):
    state = _state(request)
    try:
        state.apply_cameras(cfg)
    except ConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return state.cameras_cfg.model_dump(mode="json")


@router.get("/config/bev")
def get_bev(request: Request):
    state = _state(request)
    out = state.bev_cfg.model_dump(mode="json")
    out.update(height_px=state.bev_cfg.height_px, width_px=state.bev_cfg.width_px)
    return out


@router.post("/config/bev", dependencies=[Depends(_limiter)])
def post_bev(cfg: BevConfig, request: Request):
    state = _state(request)
    try:
        state.apply_bev(cfg)
    except ConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return get_bev(request)


@router.get("/config/scene")
def get_scene(request: Request):
    return _state(request).scene_cfg.model_dump(mode="json")


@router.post("/config/scene", dependencies=[Depends(_limiter)])
def post_scene(cfg: SceneConfig, request: Request):
    state = _state(request)
    state.apply_scene(cfg)
    return state.scene_cfg.model_dump(mode="json")


@router.post("/config/save", dependencies=[Depends(_limiter)])
def post_save(request: Request):
    return {"saved": _state(request).save_configs()}


@router.get("/debug/overlays")
def get_overlays(request: Request):
    enabled = _state(request).overlays
    return {name: name in enabled for name in OVERLAY_NAMES}


@router.post("/debug/overlays", dependencies=[Depends(_limiter)])
def post_overlays(flags: dict[str, bool], request: Request):
    state = _state(request)
    try:
        state.set_overlays({name for name, on in flags.items() if on})
    except ConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return get_overlays(request)


@router.get("/debug/projection-map")
def get_projection_map(request: Request):
    state = _state(request)
    pipeline, bev = state.pipeline, state.bev_cfg
    footprints = coverage_footprints(pipeline.weights, bev)
    total = bev.height_px * bev.width_px
    coverage_pct = {
        name: round(100.0 * float((w > 1e-3).sum()) / total, 1)
        for name, w in pipeline.weights.items()
    }
    return {
        "bev": {"x_min": bev.x_min, "x_max": bev.x_max, "y_min": bev.y_min,
                "y_max": bev.y_max, "resolution": bev.resolution,
                "height_px": bev.height_px, "width_px": bev.width_px},
        "footprints_m": footprints,
        "coverage_pct": coverage_pct,
    }


@router.get("/status")
def get_status(request: Request):
    return _state(request).stats
