#!/bin/bash
# Cloud-session setup: Blender + the official MCP for Blender addon
# (https://github.com/ahujasid/blender-mcp, package "mcp-for-blender").
# Blender runs on a virtual display (Xvfb) so the addon's socket server
# (port 9876) is up before the MCP server from .mcp.json connects to it.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

BLENDER_VERSION="5.2.2"
BLENDER_SERIES="${BLENDER_VERSION%.*}"
BLENDER_DIR="/opt/blender-${BLENDER_VERSION}-linux-x64"
ADDONS_DIR="$HOME/.config/blender/${BLENDER_SERIES}/scripts/addons"
MCP_PORT=9876
export PATH="$HOME/.local/bin:$PATH"

# 1. Blender
if [ ! -x "$BLENDER_DIR/blender" ]; then
  echo "Installing Blender ${BLENDER_VERSION}..." >&2
  tmp="$(mktemp -d)"
  curl -fsSL -o "$tmp/blender.tar.xz" \
    "https://download.blender.org/release/Blender${BLENDER_SERIES}/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
  tar -xf "$tmp/blender.tar.xz" -C /opt
  rm -rf "$tmp"
fi
ln -sf "$BLENDER_DIR/blender" /usr/local/bin/blender

# 2. Runtime libs for the GUI build on a virtual display
if ! command -v xvfb-run >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq xvfb >/dev/null
fi

# 3. uv (runs the MCP server via uvx)
if ! command -v uvx >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi

# 4. Addon, installed and enabled with the project's own CLI
if [ ! -f "$ADDONS_DIR/blender_mcp.py" ]; then
  mkdir -p "$ADDONS_DIR"
  BLENDERMCP_ADDONS_DIR="$ADDONS_DIR" uvx mcp-for-blender install-addon >&2
  blender -b --factory-startup --python-expr "
import bpy, addon_utils
addon_utils.enable('blender_mcp', default_set=True, persistent=True)
bpy.ops.wm.save_userpref()
" >/dev/null 2>&1
fi

# 5. Start Blender (with the addon's auto-started socket server) if not running
if ! (echo > "/dev/tcp/127.0.0.1/$MCP_PORT") 2>/dev/null; then
  nohup setsid xvfb-run -a -s "-screen 0 1920x1080x24" blender \
    > /tmp/blender-mcp.log 2>&1 < /dev/null &
  for _ in $(seq 1 60); do
    (echo > "/dev/tcp/127.0.0.1/$MCP_PORT") 2>/dev/null && break
    sleep 1
  done
fi

if (echo > "/dev/tcp/127.0.0.1/$MCP_PORT") 2>/dev/null; then
  echo "Blender ${BLENDER_VERSION} is running with MCP server on port $MCP_PORT" >&2
else
  echo "Blender MCP server did not start; see /tmp/blender-mcp.log" >&2
fi
