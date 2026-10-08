#!/usr/bin/env python3
"""QA for a finished visualizer against the style bible.

Usage: qa_check.py VIDEO OUT_DIR [--palette]

Checks (thresholds calibrated on the 10/10 reference vs the failed "Burning Rose"):
effective fps, change per drawing, luma jumps between drawings, black level,
pink drift, big-flash rate (WCAG 2.3.1), loop seam, palette distance.
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
    gray = cv2.cvtColor(frames.reshape(-1, frames.shape[2], 3), cv2.COLOR_BGR2GRAY).reshape(n, -1).astype(np.float32)
    mean = gray.mean(1)
    big = np.where(np.diff(mean) > 60)[0] + 1
    worst = max((np.sum((big >= s) & (big < s + fps)) for s in range(0, n)), default=0)
    seam = float(np.abs(frames[-1].astype(np.float32) - frames[0].astype(np.float32)).mean())
    typical = float(np.median(np.abs(np.diff(frames.astype(np.float32), axis=0)).mean(axis=(1, 2, 3))))
    # "drawings": first frame of each hold; a drawing must change the picture, not nudge details
    starts = np.cumsum([0] + runs[:-1])
    ch = np.array([(np.abs(gray[b] - gray[a]) > 25).mean() for a, b in zip(starts[:-1], starts[1:])])
    jump = np.abs(np.diff(mean[starts]))
    hsv = cv2.cvtColor(frames.reshape(-1, frames.shape[2], 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    h, sat, val = hsv[:, 0] * 2, hsv[:, 1], hsv[:, 2]
    reds = (sat > 90) & (val > 70)
    crimson = reds & ((h >= 345) | (h <= 12))
    pink = reds & (h >= 300) & (h < 345)
    pink_share = pink.sum() / max(1, pink.sum() + crimson.sum())
    # accents: flash / black frame / hard cut (same detector as critic_packet.py)
    small = gray.reshape(n, frames.shape[1], frames.shape[2])
    fdiff = np.abs(np.diff(small, axis=0)).mean(axis=(1, 2))
    ev = []
    for i in range(1, n):
        hit = (mean[i] > 170 >= mean[i - 1]) or (mean[i] < 8 <= mean[i - 1]) or fdiff[i - 1] > 40
        if hit and (not ev or i - ev[-1] > 8):
            ev.append(i)
    ev_rate = len(ev) / (n / fps)
    first_ev = ev[0] if ev else n
    strobe = float((fdiff > 25).mean())
    p5 = float(np.percentile(gray, 5))
    deep = float((gray < 12).mean())

    def verdict(ok):
        return "OK " if ok else "FIX"
    lines = [
        f"{verdict(12 <= eff <= 18)} effective fps {eff:.1f} (12–18); holds " +
        ", ".join(f"x{k}:{v}" for k, v in zip(u.tolist(), c.tolist())),
        f"{verdict(np.median(ch) >= 0.18)} change per drawing (median) {np.median(ch):.2f} (>= 0.18; ref 0.26, fail 0.09)",
        f"{verdict((ch < 0.05).mean() <= 0.25)} drawings with <5% change {(ch < 0.05).mean():.2f} (<= 0.25; ref 0.16, fail 0.41)",
        f"{verdict(np.median(jump) >= 5)} luma jump between drawings (median) {np.median(jump):.1f} (>= 5; ref 8.3, fail 1.6)",
        f"{verdict(p5 <= 6)} black level p5 luma {p5:.0f} (<= 6; ref 3, fail 11)",
        f"{verdict(deep >= 0.15)} share darker than 12/255 {deep:.2f} (>= 0.15; ref 0.21, fail 0.09)",
        f"{verdict(pink_share <= 0.45)} pink among reds {pink_share:.2f} (<= 0.45; ref 0.31, fail 0.85)",
        f"{verdict(ev_rate >= 0.8)} accents per second {ev_rate:.2f} (>= 0.8; ref 1.08, fail 0.40)",
        f"{verdict(first_ev <= 30)} first accent at frame {first_ev} (<= 30; ref 28, fail 141)",
        f"{verdict(strobe >= 0.18)} strobe: share of frames with big change {strobe:.2f} (>= 0.18; ref 0.25, fail 0.09)",
        f"{verdict(worst <= 3)} max big flashes in any 1 s window: {worst} (<= 3)",
        f"{verdict(seam <= max(typical * 2.5, 8))} loop seam diff {seam:.1f} vs typical frame diff {typical:.1f}",
        f"     mean luma min/median/max: {mean.min():.0f}/{np.median(mean):.0f}/{mean.max():.0f}",
        "NOTE: metrics only reject defects; they do not prove quality — run the critics (references/critics.md).",
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
            flag = "" if d[j] < 40 else "  <-- off-palette"
            if d[j] >= 40:
                bad += share
            lines.append(f"       {hx} {share * 100:5.1f}% → {CANON[j]} ({d[j]:.0f}){flag}")
        lines.insert(12, f"{verdict(bad < 0.12)} off-palette area {bad * 100:.1f}% (< 12%, distance < 40)")
    rep = "\n".join(lines)
    open(os.path.join(a.out, "qa.txt"), "w").write(rep)
    print(rep)
    print(f"contact sheets: {a.out}/sheets — review every frame")


if __name__ == "__main__":
    main()
