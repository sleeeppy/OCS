"""Start the OCS server without depending on the current working directory.

``python -m uvicorn ocs.server:app`` only works when invoked from the repo root:
it needs the root on ``sys.path`` to import ``ocs.server``, and the server mounts
``web/`` and ``workspace/`` through paths derived from the package location.

Launchers that pick their own cwd -- IDE run configs, ``.claude/launch.json``,
service wrappers -- cannot satisfy that, so this entry point fixes both itself.

    python scripts/serve.py [--host 127.0.0.1] [--port 8765] [--reload]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--reload", action="store_true")
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()

    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import uvicorn

    print(f"OCS -> http://{args.host}:{args.port}/  (root {ROOT})", flush=True)
    uvicorn.run(
        "ocs.server:app" if args.reload else _app(),
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(ROOT / "ocs")] if args.reload else None,
        log_level=args.log_level,
    )
    return 0


def _app():
    # Imported lazily so --reload can hand uvicorn the import string instead,
    # which is the only form the reloader can re-import.
    from ocs.server import app
    return app


if __name__ == "__main__":
    raise SystemExit(main())
