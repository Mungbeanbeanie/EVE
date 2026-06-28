#!/usr/bin/env bash
#
# Install EVE as a launchd LaunchAgent so it runs always-on at login with its
# native menu-bar window. Idempotent: re-running reinstalls and reloads.
#
# Usage:
#   deploy/install-agent.sh            # install + load
#   deploy/install-agent.sh --uninstall
#
set -euo pipefail

LABEL="com.eve.assistant"
EVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="$EVE_DIR/deploy/$LABEL.plist"
TARGET_DIR="$HOME/Library/LaunchAgents"
TARGET="$TARGET_DIR/$LABEL.plist"

# `launchctl bootout` is the modern unload; fall back to `unload` on older macOS.
unload() {
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null \
        || launchctl unload "$TARGET" 2>/dev/null \
        || true
}

if [[ "${1:-}" == "--uninstall" ]]; then
    unload
    rm -f "$TARGET"
    echo "✅ Uninstalled $LABEL (removed $TARGET)."
    exit 0
fi

# Prefer the project venv's Python so the agent has all deps; fall back to PATH.
PYTHON="$EVE_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3)"
    echo "⚠️  No .venv found — using $PYTHON. Run 'make setup' first for deps."
fi

mkdir -p "$TARGET_DIR" "$HOME/Library/Logs"

# Fill the template placeholders with real absolute paths.
sed -e "s#__PYTHON__#$PYTHON#g" \
    -e "s#__EVE_DIR__#$EVE_DIR#g" \
    -e "s#__HOME__#$HOME#g" \
    "$TEMPLATE" > "$TARGET"

unload
launchctl bootstrap "gui/$(id -u)" "$TARGET" 2>/dev/null \
    || launchctl load "$TARGET"

echo "✅ Installed and loaded $LABEL."
echo "   plist : $TARGET"
echo "   python: $PYTHON"
echo "   logs  : ~/Library/Logs/eve.out.log  /  eve.err.log"
echo
echo "EVE now starts at login and restarts on crash."
echo "Stop/remove it with: deploy/install-agent.sh --uninstall"
