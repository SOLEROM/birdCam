"""Pydantic schemas for cameras.yaml, bev.yaml and scene.yaml.

All external configuration enters the system through these models; nothing
downstream re-validates. Conventions (vehicle FLU frame, pitch>0 = down, ...)
are documented in bev/__init__.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Type, TypeVar, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

CAMERA_NAMES = ("front", "rear", "left", "right")


class ConfigError(ValueError):
    """Raised for any invalid or unreadable configuration."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class NoDistortion(_Frozen):
    model: Literal["none"] = "none"


class PlumbBobDistortion(_Frozen):
    """OpenCV plumb-bob radial/tangential model (k1 k2 p1 p2 k3)."""

    model: Literal["plumb_bob"] = "plumb_bob"
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0


class FisheyeDistortion(_Frozen):
    """OpenCV fisheye (equidistant) model: theta_d = theta*(1 + k1*t^2 + ...)."""

    model: Literal["fisheye"] = "fisheye"
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0


Distortion = Annotated[
    Union[NoDistortion, PlumbBobDistortion, FisheyeDistortion],
    Field(discriminator="model"),
]


class CameraConfig(_Frozen):
    width: int = Field(gt=0, le=4096)
    height: int = Field(gt=0, le=4096)
    fov_deg: float = Field(gt=0, lt=180, description="horizontal field of view")
    x: float = Field(ge=-100, le=100, description="camera X in vehicle frame [m]")
    y: float = Field(ge=-100, le=100)
    z: float = Field(gt=0, le=100, description="camera height above ground [m]")
    yaw_deg: float = Field(ge=-360, le=360, description="0 faces +X, +90 faces +Y (left)")
    pitch_deg: float = Field(ge=-90, le=90, description="positive tilts DOWN")
    roll_deg: float = Field(ge=-180, le=180)
    distortion: Distortion = NoDistortion()


class CamerasConfig(_Frozen):
    cameras: dict[str, CameraConfig]

    @model_validator(mode="after")
    def _exactly_four(self) -> "CamerasConfig":
        got = set(self.cameras)
        want = set(CAMERA_NAMES)
        if got != want:
            missing = sorted(want - got)
            extra = sorted(got - want)
            raise ValueError(
                f"cameras must be exactly {sorted(want)}; missing={missing} extra={extra}"
            )
        return self


class BevConfig(_Frozen):
    x_min: float = -5.0
    x_max: float = 8.0
    y_min: float = -5.0
    y_max: float = 5.0
    resolution: float = Field(gt=0.001, le=1.0, description="meters per BEV pixel")
    max_range: float = Field(gt=0, le=200, default=15.0)
    blend: Literal["hard", "soft"] = "soft"

    @model_validator(mode="after")
    def _extent_valid(self) -> "BevConfig":
        if self.x_min >= self.x_max:
            raise ValueError("x_min must be < x_max")
        if self.y_min >= self.y_max:
            raise ValueError("y_min must be < y_max")
        return self

    @property
    def height_px(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def width_px(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))


class ObstacleConfig(_Frozen):
    """Axis-aligned obstacle standing on the ground (z = base)."""

    type: Literal["box", "cylinder", "wall", "curb"] = "box"
    x: float
    y: float
    size_x: float = Field(gt=0, le=20, default=0.5)
    size_y: float = Field(gt=0, le=20, default=0.5)
    size_z: float = Field(gt=0, le=10, default=0.5)
    color: tuple[int, int, int] = (40, 90, 200)  # BGR


class MarkerConfig(_Frozen):
    """Flat colored square painted on the ground, centered at (x, y)."""

    x: float
    y: float
    size: float = Field(gt=0, le=10, default=1.0)
    color: tuple[int, int, int] = (255, 255, 255)  # BGR


class SceneConfig(_Frozen):
    checker_size: float = Field(gt=0.05, le=5.0, default=0.5)
    grid_step: float = Field(gt=0.1, le=10.0, default=1.0)
    obstacles: tuple[ObstacleConfig, ...] = ()
    markers: tuple[MarkerConfig, ...] = ()
    animate: bool = True


_T = TypeVar("_T", bound=BaseModel)


def load_yaml_config(path: str | Path, model: Type[_T]) -> _T:
    """Load + validate a YAML file; raises ConfigError with a clear message."""
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {p}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: top level must be a mapping")
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{p}: {exc}") from exc
