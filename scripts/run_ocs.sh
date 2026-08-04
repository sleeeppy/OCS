#!/usr/bin/env bash
# Start the OCS editor on http://127.0.0.1:8765/
#
# macOS / Linux counterpart to run_ocs.ps1. Goes through scripts/serve.py, which
# fixes cwd and sys.path itself, so this works from any directory.
#
# Usage:
#   ./scripts/run_ocs.sh [--port 8765] [--host 127.0.0.1] [--reload] [--no-browser]
set -euo pipefail

PORT=8765
HOST=127.0.0.1
RELOAD=""
OPEN_BROWSER=1

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift ;;
    --host) HOST="$2"; shift ;;
    --reload) RELOAD="--reload" ;;
    --no-browser) OPEN_BROWSER=0 ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "venv missing at $PY. Run ./scripts/setup_env.sh first." >&2
  exit 1
fi

URL="http://$HOST:$PORT/"
echo "OCS -> $URL"

if [ "$OPEN_BROWSER" -eq 1 ] && command -v open >/dev/null 2>&1; then
  # Backgrounded with a delay so the browser lands on a listening socket
  # rather than a connection refused.
  ( sleep 1.5; open "$URL" ) &
fi

exec "$PY" "$ROOT/scripts/serve.py" --host "$HOST" --port "$PORT" $RELOAD
