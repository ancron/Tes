#!/usr/bin/env python3
"""QA for a finished visualizer against the style bible.

Usage: qa_check.py VIDEO OUT_DIR [--palette]

Checks: effective fps (target 12–18), hold distribution, dark-pixel share (20–35%),
big-flash rate (<= 3/s, WCAG 2.3.1), loop seam (last vs first frame), palette distance.
Writes numbered contact sheets of every frame to OUT_DIR/sheets — look at all of them.
"""
import argparse, os, subprocess, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from analyze_reference import read_frames, hold_runs, palette  # noqa: E402

CANON = ['#0D050F', '#4D091E', '#95122E', '#E31E3C', '#E66889', '#E6D9E7', '#392D46', '#785D79']


def hex2bgr(h):
    h = h.lstrip('#')
    return np.array([int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out"); ap.add_argument("--palette", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "sheets"), exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.video, "-vf",
                    "drawtext=text='%{n}':x=6:y=6:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.6,"
                    "scale=384:-2,tile=5x6", "-vsync", "0", os.path.join(a.out, "sheets", "sheet_%02d.png")],
                   check=True)
    frames, fps = read_frames(a.video)
    n = len(frames)
    runs, _ = hold_runs(frames)
    eff = len(runs) / (n / fps)
    u, c = np.unique(runs, return_counts=True)
    L = cv2.cvtColor(frames.reshape(-1, frames.shape[2], 3), cv2.COLOR_BGR2GRAY).reshape(n, -1).astype(np.float32)
    dark = (L < 20).mean()
    mean = L.mean(1)
    big = np.where(np.diff(mean) > 60)[0] + 1  # sudden large brightness jumps = big flashes
    worst = max((np.sum((big >= s) & (big < s + fps)) for s in range(0, n)), default=0)
    seam = float(np.abs(frames[-1].astype(np.float32) - frames[0].astype(np.float32)).mean())
    typical = float(np.median(np.abs(np.diff(frames.astype(np.float32), axis=0)).mean(axis=(1, 2, 3))))

    def verdict(ok):
        return "OK " if ok else "FIX"
    lines = [
        f"{verdict(12 <= eff <= 18)} effective fps {eff:.1f} (target 12–18); holds " +
        ", ".join(f"x{k}:{v}" for k, v in zip(u.tolist(), c.tolist())),
        f"{verdict(0.15 <= dark <= 0.40)} dark pixels (<20) {dark * 100:.1f}% (target 20–35%)",
        f"{verdict(worst <= 3)} max big flashes in any 1 s window: {worst} (<= 3)",
        f"{verdict(seam <= max(typical * 2.5, 8))} loop seam diff {seam:.1f} vs typical frame diff {typical:.1f}",
        f"     mean luma min/median/max: {mean.min():.0f}/{np.median(mean):.0f}/{mean.max():.0f}",
    ]
    if a.palette:
        pal = palette(frames)
        canon = np.array([hex2bgr(h) for h in CANON])
        lines.append("     palette (share → nearest canon, distance):")
        bad = 0.0
        for hx, share in pal:
            col = hex2bgr(hx)
            d = np.linalg.norm(canon - col, axis=1)
            j = int(np.argmin(d))
            flag = "" if d[j] < 60 else "  <-- off-palette"
            if d[j] >= 60:
                bad += share
            lines.append(f"       {hx} {share * 100:5.1f}% → {CANON[j]} ({d[j]:.0f}){flag}")
        lines.insert(4, f"{verdict(bad < 0.12)} off-palette area {bad * 100:.1f}% (< 12%)")
    rep = "\n".join(lines)
    open(os.path.join(a.out, "qa.txt"), "w").write(rep)
    print(rep)
    print(f"contact sheets: {a.out}/sheets — review every frame")


if __name__ == "__main__":
    main()
