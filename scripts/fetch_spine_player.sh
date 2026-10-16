#!/usr/bin/env bash
# Download the Spine Web Player runtime into web/vendor/ for offline previews.
#
# The Spine Runtimes License Agreement restricts redistribution and requires the
# user to hold their own Spine license, so OCS does not commit this runtime -
# web/vendor/spine-player.* is gitignored and fetched here instead.
#
# Exported previews inline whatever this script leaves behind, making them work
# from file:// with no network. Skip it and previews fall back to loading the
# runtime from unpkg when opened.
#
# By running this you are asserting you hold a valid Spine license.
# See http://esotericsoftware.com/spine-runtimes-license
#
# Usage:
#   ./scripts/fetch_spine_player.sh [--version '4.2.*'] [--force]
set -euo pipefail

VERSION="4.2.*"
FORCE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/web/vendor"
mkdir -p "$VENDOR"

fetch() {
  local name="$1" url="$2" dest="$VENDOR/$1"
  if [ -f "$dest" ] && [ "$FORCE" -eq 0 ]; then
    echo "skip  $name (already present, $(( $(wc -c <"$dest") / 1024 )) KB) - use --force to refresh"
    return
  fi
  echo "fetch $name <- $url"
  # -f so an HTTP error is a failure rather than an HTML error page written to
  # disk and later inlined into every preview.
  curl -fsSL "$url" -o "$dest"
  echo "      wrote $dest ($(( $(wc -c <"$dest") / 1024 )) KB)"
}

BASE="https://unpkg.com/@esotericsoftware/spine-player@$VERSION"
fetch "spine-player.js"  "$BASE/dist/iife/spine-player.js"
fetch "spine-player.css" "$BASE/dist/spine-player.css"

echo
echo "Done. Previews exported from now on will embed the runtime."
echo "Reminder: using the Spine Runtimes requires your own Spine license."
