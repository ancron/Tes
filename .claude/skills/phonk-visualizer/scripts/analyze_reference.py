#!/usr/bin/env python3
"""Frame-by-frame breakdown of a reference edit.

Usage: analyze_reference.py VIDEO OUT_DIR [--cols 5 --rows 6]

Writes:
  OUT_DIR/sheets/sheet_XX.png  numbered contact sheets of EVERY frame (view them all)
  OUT_DIR/report.txt           effective fps, hold lengths, palette, luma curve, flashes, cuts
"""
import argparse, os, subprocess, sys
import numpy as np
import cv2


def read_frames(path, size=(256, 144)):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, size, interpolation=cv2.INTER_AREA))
    return np.array(frames), fps


def hold_runs(frames, thr=1.2):
    g = frames.mean(axis=3)
    d = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))
    runs, r = [], 1
    for same in d < thr:
        if same:
            r += 1
        else:
            runs.append(r)
            r = 1
    runs.append(r)
    return runs, d


def palette(frames, k=8):
    px = frames.reshape(-1, 3).astype(np.float32)[::7]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, lab, cent = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    cnt = np.bincount(lab.flatten(), minlength=k)
    out = []
    for i in np.argsort(-cnt):
        b, g, r = cent[i]
        out.append((f"#{int(r):02X}{int(g):02X}{int(b):02X}", cnt[i] / cnt.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "sheets"), exist_ok=True)

    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", a.video, "-vf",
        "drawtext=text='%{n}':x=6:y=6:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.6,"
        f"scale=384:-2,tile={a.cols}x{a.rows}", "-vsync", "0",
        os.path.join(a.out, "sheets", "sheet_%02d.png")], check=True)

    frames, fps = read_frames(a.video)
    n = len(frames)
    runs, diffs = hold_runs(frames)
    u, c = np.unique(runs, return_counts=True)
    luma = cv2.cvtColor(frames.reshape(-1, frames.shape[2], 3), cv2.COLOR_BGR2GRAY).reshape(n, -1)
    mean = luma.mean(1)
    flashes = [i for i in range(n) if mean[i] > 170]
    blacks = [i for i in range(n) if mean[i] < 8]
    cuts = [i + 1 for i, d in enumerate(diffs) if d > 40]

    lines = [
        f"file: {a.video}",
        f"frames: {n}  container fps: {fps:.2f}  duration: {n / fps:.2f}s",
        f"unique images: {len(runs)}  -> effective fps ~ {len(runs) / (n / fps):.1f}",
        "hold lengths (frames held: count): " + ", ".join(f"x{k}: {v}" for k, v in zip(u.tolist(), c.tolist())),
        "",
        "palette (k-means, share of area):",
        *[f"  {h}  {s * 100:5.1f}%" for h, s in palette(frames)],
        "",
        f"pixels with luma < 20: {(luma < 20).mean() * 100:.1f}%   > 235: {(luma > 235).mean() * 100:.1f}%",
        f"flash frames (mean luma > 170): {flashes}",
        f"black frames (mean luma < 8): {blacks}",
        f"hard cuts (mean abs diff > 40): {cuts}",
        "",
        "mean luma per frame:",
        " ".join(str(int(x)) for x in mean),
    ]
    rep = "\n".join(lines)
    open(os.path.join(a.out, "report.txt"), "w").write(rep)
    print(rep)
    print(f"\nContact sheets: {a.out}/sheets/ — open every one of them.")


if __name__ == "__main__":
    sys.exit(main())
