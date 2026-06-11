#!/usr/bin/env bash
# Install everything needed to run bev_web_sim on a fresh Linux host.
# Usage: ./install.sh [--with-webots] [--with-e2e]
#   --with-webots   also install the Webots simulator (snap, needs sudo)
#   --with-e2e      also install Playwright + Chromium for browser E2E tests
set -euo pipefail
cd "$(dirname "$0")"

WITH_WEBOTS=0
WITH_E2E=0
for arg in "$@"; do
  case "$arg" in
    --with-webots) WITH_WEBOTS=1 ;;
    --with-e2e) WITH_E2E=1 ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

echo "==> apt prerequisites (python3-venv, pip, git)"
if ! dpkg -s python3-venv >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y python3-venv python3-pip git
fi

echo "==> python virtualenv + dependencies"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q

if [ "$WITH_WEBOTS" = 1 ]; then
  if ! command -v webots >/dev/null 2>&1; then
    echo "==> installing Webots (snap, requires sudo)"
    sudo snap install webots
  else
    echo "==> Webots already installed: $(command -v webots)"
  fi
fi

if [ "$WITH_E2E" = 1 ]; then
  echo "==> Playwright + Chromium for E2E tests"
  .venv/bin/pip install -q playwright pytest-playwright
  .venv/bin/playwright install chromium
fi

echo "==> generating Webots world from configs"
.venv/bin/python scripts/gen_webots.py

echo "==> running the fast test suite"
.venv/bin/python -m pytest -q -m "not webots and not e2e and not perf"

echo
echo "Install complete."
echo "  ./run.sh            # dashboard with the synthetic source (no simulator)"
echo "  ./run.sh webots     # dashboard fed by live Webots cameras"
echo "  http://127.0.0.1:8000"
