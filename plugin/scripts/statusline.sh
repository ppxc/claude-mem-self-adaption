#!/bin/bash
# Claude Code Status Line for claude-mem
# Usage: Configure in .claude/settings.local.json:
# { "statusLine": { "type": "command", "command": "bash \"$HOME/.claude/statusline.sh\"" } }

# Detect platform
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$(uname -s)" == "Windows_NT" ]]; then
  PLATFORM="windows"
  NODE_BIN="node"
else
  PLATFORM="unix"
  NODE_BIN="node"
fi

# Path to the statusline script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATUSLINE_SCRIPT="$SCRIPT_DIR/statusline-model.js"

# Run the Node.js script
exec "$NODE_BIN" "$STATUSLINE_SCRIPT" "$@"
