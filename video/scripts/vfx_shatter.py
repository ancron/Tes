#!/usr/bin/env python3
"""Build VFX elements from real footage (rendered offline, composited in Remotion).

  shatter: a graded rose frame breaks into Voronoi shards textured with the rose itself,
           thrown out as two wings with drag, spin, sub-frame motion blur, white-hot edges
           cooling to crimson, plus a light core.
  inhale:  embers and small shards spiral INTO the centre (the breath before the drop).

Usage: python3 scripts/vfx_shatter.py shatter|inhale  → public/footage/graded/vfx_<name>.mp4
"""
import os
import subprocess
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.join(os.path.dirname(__file__), '..', 'public', 'footage', 'graded')
W, H = 1280, 720          # simulate at 720p, upscale to 1080p on output
OW, OH = 1920, 1080
FPS = 30
rng = np.random.default_rng(7)


def grab(clip, t):
    out = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(t), '-i', os.path.join(ROOT, f'{clip}.mp4'),
                          '-frames:v', '1', '-vf', f'scale={W}:{H}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                         capture_output=True).stdout
    return np.frombuffer(out, np.uint8).reshape(H, W, 3).astype(np.float32) / 255


def make_shards(img, n_small, n_big, radius):
    cx, cy = W / 2, H / 2
    pts = []
    for n, rmax, pw in ((n_big, radius * 1.15, 0.9), (n_small, radius * 0.75, 0.55)):
        r = rmax * rng.random(n) ** pw
        a = rng.uniform(0, 2 * np.pi, n)
        pts.append(np.stack([cx + r * np.cos(a), cy + r * np.sin(a) * 0.8], 1))
    pts = np.concatenate(pts)
    yy, xx = np.mgrid[0:H, 0:W]
    _, lab = cKDTree(pts).query(np.stack([xx.ravel(), yy.ravel()], 1))
    lab = lab.reshape(H, W)
    # only the rose region shatters; outside stays as the dark world
    region = ((xx - cx) ** 2 + ((yy - cy) / 0.8) ** 2) < radius ** 2
    shards = []
    for i in range(len(pts)):
        m = (lab == i) & region
        if m.sum() < 30:
            continue
        ys, xs = np.nonzero(m)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        mask = m[y0:y1, x0:x1].astype(np.float32)
        mask = cv2.GaussianBlur(mask, (3, 3), 0.7)           # soft, slightly ragged edge
        rgba = np.dstack([img[y0:y1, x0:x1], mask])
        edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        shards.append(dict(rgba=rgba, edge=edge, c=np.array([(x0 + x1) / 2, (y0 + y1) / 2]), area=m.sum(),
                           off=np.array([x0, y0], np.float32)))
    return shards


def blit(canvas, alpha_acc, rgba, edge, cx, cy, ang, scl, heat, w):
    h0, w0 = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), np.degrees(ang), scl)
    corners = np.array([[0, 0, 1], [w0, 0, 1], [0, h0, 1], [w0, h0, 1]], np.float32) @ M.T
    bw = int(np.ceil(corners[:, 0].max() - corners[:, 0].min())) + 2
    bh = int(np.ceil(corners[:, 1].max() - corners[:, 1].min())) + 2
    M[0, 2] += -corners[:, 0].min() + 1
    M[1, 2] += -corners[:, 1].min() + 1
    tex = cv2.warpAffine(rgba, M, (bw, bh), flags=cv2.INTER_LINEAR)
    e = cv2.warpAffine(edge, M, (bw, bh), flags=cv2.INTER_LINEAR)
    x0, y0 = int(cx - bw / 2), int(cy - bh / 2)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + bw), min(H, y0 + bh)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    t = tex[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
    ee = e[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0, None]
    a = t[..., 3:4] * w
    # heat burns RED (never toward pink): red channel boosted, edges white-hot -> crimson as they cool
    col = t[..., :3] * (0.8 + 0.4 * heat) + heat * np.array([0.45, 0.03, 0.05]) + \
        ee * (np.array([1.0, 0.9, 0.88]) * 1.4 * heat + np.array([0.9, 0.08, 0.14]) * 0.9)
    canvas[dy0:dy1, dx0:dx1] += col * a
    alpha_acc[dy0:dy1, dx0:dx1] += a


def bloom(img):
    hot = np.clip(img - 0.6, 0, None)
    b = cv2.GaussianBlur(hot, (0, 0), 6) * 0.8 + cv2.GaussianBlur(hot, (0, 0), 22) * 0.6
    tint = np.array([1.0, 0.16, 0.24], np.float32)
    return img + b * tint * 1.0


def encode(frames, name):
    dst = os.path.join(ROOT, f'vfx_{name}.mp4')
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{OW}x{OH}',
                          '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '12', '-preset', 'medium',
                          '-pix_fmt', 'yuv420p', '-g', '10', dst], stdin=subprocess.PIPE)
    for f in frames:
        f = 1 - np.exp(-np.clip(f, 0, None) * 1.3)
        f = f / (1 - np.exp(-1.3))
        u8 = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        p.stdin.write(cv2.resize(u8, (OW, OH), interpolation=cv2.INTER_LANCZOS4).tobytes())
    p.stdin.close()
    p.wait()
    print('wrote', dst)


def shatter(n_frames=20):
    img = grab('171', 3.0)
    img = np.clip(img * 1.15, 0, 1)
    shards = make_shards(img, 520, 90, radius=0.42 * W)
    cx, cy = W / 2, H / 2
    for s in shards:
        d = s['c'] - [cx, cy]
        dist = np.linalg.norm(d) / (0.42 * W)
        th = np.arctan2(d[1], d[0])
        # wings: squash vertical component, bias upward, outer & small shards faster
        dirv = np.array([np.cos(th), np.sin(th) * 0.38 - 0.22])
        dirv /= np.linalg.norm(dirv) + 1e-6
        small = 1.0 / (1 + s['area'] / 900)
        s['v'] = dirv * (380 + 1100 * rng.random() * (0.35 + dist) + 900 * small)
        s['w'] = rng.normal(0, 5)
        s['z'] = rng.uniform(-0.2, 1.1) * (1.4 if s['area'] > 2500 else 0.6)   # some big pieces fly at camera
        s['k'] = rng.uniform(1.6, 3.0)
    frames = []
    for i in range(n_frames):
        canvas = np.zeros((H, W, 3), np.float32)
        acc = np.zeros((H, W, 1), np.float32)
        subs = 12
        for k in range(subs):
            t = (i + 2 + k / subs * 0.9) / FPS          # start 2 frames in (the flash covers the hit)
            for s in shards:
                disp = s['v'] * (1 - np.exp(-s['k'] * t)) / s['k']
                c = s['c'] + disp
                heat = np.exp(-t * 3.2)
                blit(canvas, acc, s['rgba'], s['edge'], c[0], c[1], s['w'] * t, 1 + s['z'] * t, heat, 1 / subs)
        out = canvas / np.maximum(acc, 1e-3) * np.clip(acc, 0, 1)
        # light core where the rose was
        t = i / FPS
        yy, xx = np.mgrid[0:H, 0:W]
        rr = np.sqrt((xx - cx) ** 2 + ((yy - cy) * 1.4) ** 2) / W
        core = np.exp(-rr / (0.02 + 0.05 * t)) * np.exp(-t * 9)
        out = out + core[..., None] * np.array([1.0, 0.55, 0.55]) * 1.6
        frames.append(bloom(out))
        print('shatter frame', i, flush=True)
    encode(frames, 'shatter')


def inhale(n_frames=14):
    img = grab('100899', 2.0)
    shards = make_shards(np.clip(img * 1.3, 0, 1), 260, 0, radius=0.5 * W)
    cx, cy = W / 2, H / 2
    for s in shards:
        d = s['c'] - [cx, cy]
        s['r0'] = np.linalg.norm(d) * rng.uniform(1.6, 2.8)
        s['a0'] = np.arctan2(d[1], d[0])
        s['spin'] = rng.uniform(2.5, 4.5)
        s['w'] = rng.normal(0, 4)
    frames = []
    T = n_frames / FPS
    for i in range(n_frames):
        canvas = np.zeros((H, W, 3), np.float32)
        acc = np.zeros((H, W, 1), np.float32)
        subs = 12
        for k in range(subs):
            t = (i + k / subs * 0.9) / FPS
            p = (t / T) ** 1.6                               # accelerating pull
            for s in shards:
                r = s['r0'] * (1 - p) + 4
                a = s['a0'] + s['spin'] * p
                c = np.array([cx + r * np.cos(a), cy + r * np.sin(a) * 0.75])
                blit(canvas, acc, s['rgba'], s['edge'], c[0], c[1], s['w'] * t, 0.35 + 0.65 * (1 - p),
                     0.2 + 0.8 * p, 1 / subs)
        out = canvas / np.maximum(acc, 1e-3) * np.clip(acc, 0, 1)
        out *= 0.35 + 0.65 * (i / n_frames)
        frames.append(bloom(out))
        print('inhale frame', i, flush=True)
    encode(frames, 'inhale')


if __name__ == '__main__':
    {'shatter': shatter, 'inhale': inhale}[sys.argv[1]]()
