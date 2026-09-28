# Vendored assets

xterm.js renders the raw ANSI `pipe-pane` captured, so a log looks in the
browser the way it looked in the terminal.
It is vendored rather than fetched so the dashboard works with no network and
cannot break because someone else's CDN did.

| File | Package | Version | License |
|---|---|---|---|
| `xterm.js` | `@xterm/xterm` | 5.5.0 | MIT (`LICENSE.xterm`) |
| `xterm.css` | `@xterm/xterm` | 5.5.0 | MIT (`LICENSE.xterm`) |
| `xterm-addon-fit.js` | `@xterm/addon-fit` | 0.10.0 | MIT (`LICENSE.xterm`) |

Refresh with the `curl` lines in `scripts/vendor-xterm.sh`.
