#!/usr/bin/env bash
# Re-vendor the dashboard's terminal renderer. No build pipeline: the
# published UMD bundles are used exactly as shipped.
set -euo pipefail
XTERM=5.5.0
FIT=0.10.0
dest="$(cd "$(dirname "$0")/../buddy/web/static/vendor" && pwd)"
curl -sSfL -o "$dest/xterm.js"            "https://cdn.jsdelivr.net/npm/@xterm/xterm@$XTERM/lib/xterm.js"
curl -sSfL -o "$dest/xterm.css"           "https://cdn.jsdelivr.net/npm/@xterm/xterm@$XTERM/css/xterm.css"
curl -sSfL -o "$dest/xterm-addon-fit.js"  "https://cdn.jsdelivr.net/npm/@xterm/addon-fit@$FIT/lib/addon-fit.js"
curl -sSfL -o "$dest/LICENSE.xterm"       "https://raw.githubusercontent.com/xtermjs/xterm.js/master/LICENSE"
echo "vendored xterm.js $XTERM and addon-fit $FIT into $dest"
