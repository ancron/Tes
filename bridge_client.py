#!/usr/bin/env python3
"""
ntfy-bridge client.

Sends HTTP jobs to the bridge server via ntfy and waits for responses.
Behaves like a minimal curl wrapper.

Usage:
    python bridge_client.py [OPTIONS] <url>
    python bridge_client.py -X POST -H "Content-Type: application/json" -d '{"a":1}' https://example.com

Options:
    -X <method>       HTTP method (default: GET)
    -H <header>       Header in "Key: Value" form (repeatable)
    -d <body>         Request body
    --timeout <sec>   Max wait for response (default: 60)
    -v                Verbose: print status and response headers too

Environment variables:
    NTFY_BASE         ntfy server URL (default: https://ntfy.sh)
    BRIDGE_TOPIC      request topic (required)
    BRIDGE_TOKEN      shared secret (must match server)
"""

import json
import os
import sys
import threading
import time
import uuid

import requests


NTFY_BASE = os.environ.get("NTFY_BASE", "https://ntfy.sh").rstrip("/")
BRIDGE_TOPIC = os.environ.get("BRIDGE_TOPIC", "")
BRIDGE_TOKEN = os.environ.get("BRIDGE_TOKEN", "")


def parse_args(argv):
    args = {
        "method": None,
        "url": None,
        "headers": {},
        "body": None,
        "timeout": 60,
        "verbose": False,
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-X":
            i += 1
            args["method"] = argv[i]
        elif a == "-H":
            i += 1
            k, _, v = argv[i].partition(":")
            args["headers"][k.strip()] = v.strip()
        elif a == "-d":
            i += 1
            args["body"] = argv[i]
        elif a == "--timeout":
            i += 1
            args["timeout"] = int(argv[i])
        elif a == "-v":
            args["verbose"] = True
        elif not a.startswith("-"):
            if args["url"] is None:
                args["url"] = a
            elif args["method"] is None:
                args["method"] = a
        i += 1

    if args["method"] is None:
        args["method"] = "POST" if args["body"] else "GET"
    return args


def send_job(args: dict) -> dict | None:
    if not BRIDGE_TOPIC:
        print("Error: BRIDGE_TOPIC not set", file=sys.stderr)
        sys.exit(1)

    req_id = uuid.uuid4().hex
    reply_topic = f"{BRIDGE_TOPIC}-r-{req_id[:12]}"

    job = {
        "id": req_id,
        "reply_topic": reply_topic,
        "method": args["method"].upper(),
        "url": args["url"],
        "headers": args["headers"],
        "body": args["body"],
    }
    if BRIDGE_TOKEN:
        job["token"] = BRIDGE_TOKEN

    response_holder = [None]
    got_response = threading.Event()

    def listen():
        try:
            with requests.get(
                f"{NTFY_BASE}/{reply_topic}/sse",
                headers={"Accept": "text/event-stream"},
                stream=True,
                timeout=None,
            ) as resp:
                for raw_line in resp.iter_lines():
                    if got_response.is_set():
                        break
                    if not raw_line:
                        continue
                    line = raw_line.decode() if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data: "):
                        continue
                    try:
                        envelope = json.loads(line[6:])
                        msg_text = envelope.get("message", "")
                        if not msg_text:
                            continue
                        result = json.loads(msg_text)
                        if result.get("id") == req_id:
                            response_holder[0] = result
                            got_response.set()
                            return
                    except json.JSONDecodeError:
                        pass
        except Exception as exc:
            print(f"Listener error: {exc}", file=sys.stderr)

    listener = threading.Thread(target=listen, daemon=True)
    listener.start()
    time.sleep(0.8)  # let subscriber register before we publish

    try:
        requests.post(
            f"{NTFY_BASE}/{BRIDGE_TOPIC}",
            data=json.dumps(job),
            headers={"Content-Type": "text/plain"},
            timeout=15,
        )
    except Exception as exc:
        print(f"Failed to submit job: {exc}", file=sys.stderr)
        sys.exit(1)

    got_response.wait(timeout=args["timeout"])

    if not got_response.is_set():
        print("Timeout: no response from bridge", file=sys.stderr)
        sys.exit(124)

    return response_holder[0]


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    args = parse_args(sys.argv[1:])
    if not args["url"]:
        print("Error: URL required", file=sys.stderr)
        sys.exit(1)

    result = send_job(args)
    if result is None:
        sys.exit(1)

    if not result.get("ok"):
        print(f"Bridge error: {result.get('error')}", file=sys.stderr)
        sys.exit(1)

    if args["verbose"]:
        print(f"HTTP {result['status']}", file=sys.stderr)
        for k, v in (result.get("headers") or {}).items():
            print(f"< {k}: {v}", file=sys.stderr)
        print(file=sys.stderr)

    body = result.get("body", "")
    print(body, end="")

    sys.exit(0 if result["status"] < 400 else 1)


if __name__ == "__main__":
    main()
