#!/usr/bin/env bash
# Spin up the BEV demo.
# Usage:
#   ./run.sh                 # synthetic source (no simulator needed)
#   ./run.sh webots          # launch Webots + dashboard fed by its cameras
#   ./run.sh synthetic 8080  # optional port as 2nd arg
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-synthetic}"
PORT="${2:-8000}"
WEBOTS_PORT=1234

if [ ! -d .venv ]; then
  echo "no .venv found — run ./install.sh first" >&2
  exit 1
fi

cleanup() {
  if [ -n "${WEBOTS_PID:-}" ] && kill -0 "$WEBOTS_PID" 2>/dev/null; then
    kill "$WEBOTS_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

case "$MODE" in
  synthetic)
    echo "==> dashboard (synthetic source): http://127.0.0.1:$PORT"
    exec .venv/bin/python -m web --source synthetic --port "$PORT"
    ;;
  webots)
    command -v webots >/dev/null 2>&1 || {
      echo "webots not installed — run ./install.sh --with-webots" >&2; exit 1; }
    echo "==> regenerating Webots world from configs"
    .venv/bin/python scripts/gen_webots.py
    WORLD="$PWD/webots/worlds/bev_test_world.wbt"
    # snap-confined webots can only read under $HOME: stage the assets there
    if command -v webots | grep -q "^/snap/"; then
      STAGE="$HOME/.cache/bev_web_sim"
      mkdir -p "$STAGE"
      cp -r webots "$STAGE/"
      WORLD="$STAGE/webots/worlds/bev_test_world.wbt"
      echo "==> staged world for snap confinement: $WORLD"
    fi
    echo "==> launching Webots"
    WEBOTS_LOG="$(mktemp /tmp/bev_webots_XXXX.log)"
    webots --batch --mode=realtime --port="$WEBOTS_PORT" --extern-urls \
      "$WORLD" > "$WEBOTS_LOG" 2>&1 &
    WEBOTS_PID=$!
    # webots bumps the port when busy: parse the announced extern URL
    ACTUAL_PORT=""
    for _ in $(seq 1 120); do
      ACTUAL_PORT=$(grep -oE 'ipc://[0-9]+/rover' "$WEBOTS_LOG" | head -1 | grep -oE '[0-9]+' || true)
      [ -n "$ACTUAL_PORT" ] && break
      kill -0 "$WEBOTS_PID" 2>/dev/null || { tail -20 "$WEBOTS_LOG" >&2; echo "webots died" >&2; exit 1; }
      sleep 1
    done
    [ -n "$ACTUAL_PORT" ] || { echo "webots never announced its controller URL" >&2; exit 1; }
    export WEBOTS_CONTROLLER_URL="tcp://localhost:$ACTUAL_PORT/rover"
    echo "==> webots controller: $WEBOTS_CONTROLLER_URL"
    echo "==> dashboard (webots source): http://127.0.0.1:$PORT"
    .venv/bin/python -m web --source webots --port "$PORT"
    ;;
  folder)
    FOLDER="${2:?usage: ./run.sh folder <image-dir> [port]}"
    PORT="${3:-8000}"
    echo "==> dashboard (folder replay): http://127.0.0.1:$PORT"
    exec .venv/bin/python -m web --source folder --folder "$FOLDER" --port "$PORT"
    ;;
  *)
    echo "usage: ./run.sh [synthetic|webots|folder] ..." >&2
    exit 1
    ;;
esac
