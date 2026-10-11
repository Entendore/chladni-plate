#!/usr/bin/env bash
# Launcher: prefers the project-local .venv if present.
cd "$(dirname "$0")"
PY=python
[[ -x ".venv/bin/python" ]] && PY=".venv/bin/python"
exec "$PY" chladni_plate.py "$@"
