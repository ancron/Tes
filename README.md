# Tes

## Blender + MCP in Claude Code cloud sessions

- `.claude/hooks/session-start.sh` installs Blender 5.2.2 and the
  [MCP for Blender](https://github.com/ahujasid/blender-mcp) addon, then starts
  Blender on a virtual display (Xvfb) with the addon's server on port 9876.
- `.mcp.json` registers the MCP server (`uvx mcp-for-blender`, telemetry off).
- Log of the background Blender: `/tmp/blender-mcp.log`.
