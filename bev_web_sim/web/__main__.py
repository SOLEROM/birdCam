"""Entry point: python -m web --source synthetic|webots|folder"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from web.app import create_app
from web.runtime import AppSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="BEV web simulation dashboard")
    parser.add_argument("--source", default="synthetic",
                        choices=["synthetic", "webots", "folder"])
    parser.add_argument("--config-dir", default="config", type=Path)
    parser.add_argument("--folder", default=None, type=Path,
                        help="image folder for --source folder")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--fps", default=12.0, type=float)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    app = create_app(AppSettings(
        config_dir=args.config_dir, source=args.source, folder=args.folder, fps=args.fps,
    ))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
