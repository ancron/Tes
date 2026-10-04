#!/usr/bin/env python3
"""
ntfy-bridge server.

Subscribes to a ntfy topic (outbound SSE connection), reads HTTP job descriptions,
executes them, and publishes responses back to ntfy.

Environment variables:
  NTFY_BASE         ntfy server URL (default: https://ntfy.sh)
  BRIDGE_TOPIC      topic to listen for incoming jobs (required)
  BRIDGE_TOKEN      shared secret — jobs without this token are ignored
  MAX_WORKERS       max parallel jobs (default: 8)
"""

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

NTFY_BASE = os.environ.get("NTFY_BASE", "https://ntfy.sh").rstrip("/")
BRIDGE_TOPIC = os.environ.get("BRIDGE_TOPIC", "")
BRIDGE_TOKEN = os.environ.get("BRIDGE_TOKEN", "")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))

BLOCKED_PREFIXES = ("file://", "ftp://", "gopher://")
BLOCKED_HOSTS = ("localhost", "127.", "10.", "192.168.", "169.254.", "::1")


def is_safe_url(url: str) -> bool:
    for prefix in BLOCKED_PREFIXES:
        if url.lower().startswith(prefix):
            return False
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        for blocked in BLOCKED_HOSTS:
            if host.startswith(blocked) or host == blocked.rstrip("."):
                return False
    except Exception:
        return False
    return True


def handle_job(job: dict):
    req_id = job.get("id", "?")
    reply_topic = job.get("reply_topic", "")
    method = job.get("method", "GET").upper()
    url = job.get("url", "")
    headers = job.get("headers") or {}
    body = job.get("body")

    if not url:
        log.warning("[%s] No URL in job, skipping", req_id)
        return
    if not reply_topic:
        log.warning("[%s] No reply_topic, skipping", req_id)
        return
    if not is_safe_url(url):
        log.warning("[%s] Blocked URL: %s", req_id, url)
        _publish(reply_topic, {"id": req_id, "ok": False, "error": "URL blocked"})
        return

    log.info("[%s] %s %s", req_id, method, url)

    try:
        resp = requests.request(
            method,
            url,
            headers=headers,
            data=body.encode() if isinstance(body, str) else body,
            timeout=30,
            allow_redirects=True,
            verify=True,
        )
        result = {
            "id": req_id,
            "ok": True,
            "status": resp.status_code,
            "headers": dict(resp.headers),
            "body": resp.text,
        }
        log.info("[%s] → %d (%d bytes)", req_id, resp.status_code, len(resp.content))
    except Exception as exc:
        result = {"id": req_id, "ok": False, "error": str(exc)}
        log.error("[%s] Request failed: %s", req_id, exc)

    _publish(reply_topic, result)


def _publish(topic: str, payload: dict):
    try:
        requests.post(
            f"{NTFY_BASE}/{topic}",
            data=json.dumps(payload),
            headers={"Content-Type": "text/plain"},
            timeout=15,
        )
    except Exception as exc:
        log.error("Failed to publish response: %s", exc)


def run_bridge():
    if not BRIDGE_TOPIC:
        log.error("BRIDGE_TOPIC not set")
        sys.exit(1)

    log.info("Bridge started. Topic: %s/%s", NTFY_BASE, BRIDGE_TOPIC)
    if BRIDGE_TOKEN:
        log.info("Token auth: enabled")
    else:
        log.warning("Token auth: DISABLED — anyone can submit jobs")

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    while True:
        try:
            log.info("Connecting to SSE stream…")
            with requests.get(
                f"{NTFY_BASE}/{BRIDGE_TOPIC}/sse",
                headers={"Accept": "text/event-stream"},
                stream=True,
                timeout=None,
            ) as resp:
                resp.raise_for_status()
                log.info("SSE stream connected")
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    try:
                        envelope = json.loads(data_str)
                        msg_text = envelope.get("message", "")
                        if not msg_text:
                            continue
                        job = json.loads(msg_text)
                        if BRIDGE_TOKEN and job.get("token") != BRIDGE_TOKEN:
                            log.warning("Job rejected: wrong token")
                            continue
                        executor.submit(handle_job, job)
                    except json.JSONDecodeError:
                        pass
        except Exception as exc:
            log.error("SSE error: %s — retrying in 5 s", exc)
            time.sleep(5)


if __name__ == "__main__":
    run_bridge()
