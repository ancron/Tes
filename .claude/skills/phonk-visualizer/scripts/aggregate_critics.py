#!/usr/bin/env python3
"""Aggregate critic JSONs into a verdict (see references/critics.md).

Usage: aggregate_critics.py PACKET_DIR CRITIC_JSON [CRITIC_JSON ...]

Reads PACKET_DIR/.key.json to un-blind X/Y, drops critics that failed calibration
(scored the known-bad clip more than 1.5 above that role's baseline from
references/calibration/baseline.json), and applies the gate: mean >= 8 and min >= 7.
"""
import json
import os
import re
import sys

BASELINE = os.path.join(os.path.dirname(__file__), '..', 'references', 'calibration', 'baseline.json')


def load(path):
    txt = open(path).read()
    m = re.search(r'\{.*\}', txt, re.S)   # tolerate prose around the JSON
    return json.loads(m.group(0))


def main():
    packet, files = sys.argv[1], sys.argv[2:]
    key = json.load(open(f'{packet}/.key.json'))
    cand, cal = key['candidate'], key.get('calibration')
    base = json.load(open(BASELINE))['scores_by_role'] if os.path.exists(BASELINE) else {}
    valid, dropped = [], []
    for f in files:
        c = load(f)
        sc = c.get('scores', {})
        limit = base.get(c.get('role'), 4) + 1.5
        if cal and float(sc.get(cal, 0)) > limit:
            dropped.append((c.get('role', f), sc.get(cal), limit))
            continue
        valid.append(c)
    print(f'candidate = {cand}, calibration = {cal}')
    for role, s, limit in dropped:
        print(f'  DROPPED {role}: scored the known-bad clip {s} (> {limit}) — rerun this role')
    if not valid:
        print('no valid critics')
        return 1
    scores = [(c.get('role', '?'), float(c['scores'][cand])) for c in valid]
    for role, s in sorted(scores, key=lambda x: x[1]):
        print(f'  {role:12s} {s:4.1f}')
    vals = [s for _, s in scores]
    mean, low = sum(vals) / len(vals), min(vals)
    ok = mean >= 8 and low >= 7 and not dropped
    print(f'mean {mean:.1f}  min {low:.1f}  ->  {"PASS: show the user" if ok else "FAIL: fix and rerun"}')
    print('\ntop fixes (lowest-scoring roles first):')
    for c in sorted(valid, key=lambda c: float(c['scores'][cand])):
        for fx in c.get('fixes', {}).get(cand, [])[:3]:
            print(f'  [{c.get("role")}] {fx}')
    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
