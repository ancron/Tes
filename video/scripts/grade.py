#!/usr/bin/env python3
"""Grade stock clips into the edit's palette (luma → gradient map) as 1920x1080 @30fps.

Usage:
  python3 scripts/grade.py preview [ID ...]   # sample frames → scratch sheet for tuning
  python3 scripts/grade.py encode  [ID ...]   # writes public/footage/graded/<ID>.mp4

Why offline: the palette (true black, crimson — not pink — white cores) must be exact,
and baking the bloom here keeps the Remotion render light (CPU-only WebGL).
"""
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..', 'public', 'footage')
W, H, FPS = 1920, 1080, 30

RED = [(0.00, '#000000'), (0.14, '#040102'), (0.30, '#3A0716'), (0.46, '#8A1029'), (0.62, '#CF1532'),
       (0.80, '#EE1B36'), (0.90, '#FF3E52'), (0.97, '#FFD2CE'), (1.00, '#FFF6F4')]
REDC = [(0.00, '#000000'), (0.10, '#05060E'), (0.24, '#14152A'), (0.38, '#4A0F26'), (0.52, '#9E142E'),
        (0.68, '#E31E3C'), (0.84, '#FF4256'), (0.95, '#FFE2DE'), (1.00, '#FFFFFF')]
COLD = [(0.00, '#000000'), (0.20, '#06040B'), (0.45, '#2A2340'), (0.70, '#7B6E93'), (0.88, '#D2C8E2'),
        (1.00, '#F6F2FF')]

# per-clip grading: channel weights for intensity, levels (lo, hi, gamma), invert, palette, bloom, crop (x,y,w,h in source px)
CFG = {
    '171':    dict(w=(0.75, 0.15, 0.10), lo=0.10, hi=0.92, gamma=1.30, pal=RED, bloom=0.55),
    '100917': dict(w=(0.85, 0.10, 0.05), lo=0.08, hi=0.80, gamma=1.25, pal=RED, bloom=0.5, crop=(0, 250, 720, 405)),
    '100899': dict(w=(0.85, 0.10, 0.05), lo=0.10, hi=0.85, gamma=1.30, pal=RED, bloom=0.6),
    '100898': dict(w=(0.85, 0.10, 0.05), lo=0.10, hi=0.85, gamma=1.30, pal=RED, bloom=0.5),
    '52304':  dict(w=(0.30, 0.55, 0.15), lo=0.06, hi=1.00, gamma=1.45, pal=RED, bloom=0.7),
    '52312':  dict(w=(0.30, 0.55, 0.15), lo=0.06, hi=1.00, gamma=1.45, pal=RED, bloom=0.7),
    '3759':   dict(w=(0.30, 0.55, 0.15), lo=0.05, hi=0.95, gamma=1.25, pal=RED, bloom=0.8),
    '50951':  dict(w=(0.33, 0.34, 0.33), lo=0.08, hi=0.90, gamma=1.20, pal=RED, bloom=0.4),
    '41999':  dict(w=(1.00, -1.00, 0.00), lo=0.10, hi=1.10, gamma=1.6, pal=RED, bloom=0.5),
    '3465':   dict(w=(0.40, 0.45, 0.15), lo=0.03, hi=0.70, gamma=0.90, pal=RED, bloom=1.0),
    '4426':   dict(w=(0.40, 0.45, 0.15), lo=0.05, hi=0.75, gamma=1.00, pal=RED, bloom=0.9),
    '3463':   dict(w=(0.40, 0.45, 0.15), lo=0.03, hi=0.70, gamma=0.90, pal=RED, bloom=1.0, crop=(0, 420, 1080, 608)),
    '1038':   dict(w=(0.33, 0.34, 0.33), lo=0.04, hi=0.75, gamma=1.00, pal=REDC, bloom=0.6),
    '40938':  dict(w=(0.33, 0.34, 0.33), lo=0.18, hi=1.20, gamma=1.30, pal=RED, bloom=0.6, invert=True),
    '33899':  dict(w=(0.40, 0.30, 0.30), lo=0.06, hi=0.75, gamma=1.10, pal=REDC, bloom=0.5),
}
# the "cold hands" scene uses a second grade of 40938
CFG['40938c'] = dict(CFG['40938'], hi=0.80, pal=COLD, src='40938')
CFG['40938f'] = dict(CFG['40938'], src='40938', burn='red')     # crimson hands made of fire
CFG['40938cf'] = dict(CFG['40938'], src='40938', burn='cold')   # white-hot hands, crimson flames on the edge


def lut(stops, n=1024):
    pos = np.array([p for p, _ in stops], np.float32)
    cols = np.array([[int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] for _, c in stops], np.float32)
    xs = np.linspace(0, 1, n, dtype=np.float32)
    return np.stack([np.interp(xs, pos, cols[:, k]) for k in range(3)], 1)


def grade(rgb, c):
    f = rgb.astype(np.float32) / 255
    w = np.array(c['w'], np.float32)
    v = f @ w
    if c.get('invert'):
        v = 1 - v
    v = np.clip((v - c['lo']) / (c['hi'] - c['lo']), 0, 1) ** c['gamma']
    L = lut(c['pal'])
    out = L[(v * 1023).astype(np.int32)]
    if c.get('bloom'):
        hot = np.clip(v - 0.62, 0, 1)
        small = cv2.resize(hot, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        b = cv2.GaussianBlur(small, (0, 0), 9) + cv2.GaussianBlur(small, (0, 0), 30) * 0.6
        b = cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
        tint = np.array([0.95, 0.10, 0.18], np.float32) if c['pal'] is RED else np.array([0.75, 0.70, 0.95], np.float32)
        out = out + b * tint * c['bloom'] * 1.6
    out = 1 - np.exp(-out * 1.15)          # soft shoulder: no clipped mush
    out = out / (1 - np.exp(-1.15))
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def burn_hands(rgb, fire_rgb, mode):
    f = rgb.astype(np.float32) / 255
    v = 1 - (f @ np.array([0.33, 0.34, 0.33], np.float32))
    hand = np.clip((v - 0.35) / 0.25, 0, 1)                       # 1 inside the silhouette
    fire = fire_rgb.astype(np.float32) / 255 @ np.array([0.3, 0.55, 0.15], np.float32)
    fire = np.clip((fire - 0.08) / 0.85, 0, 1) ** 1.2
    # flames rise off the contour: smear the hand mask upward, keep only the outside rim
    up = hand.copy()
    for k in range(1, 9):
        up = np.maximum(up, np.roll(hand, -k * 9, axis=0) * (1 - k / 9))
    rim = np.clip(up - hand, 0, 1)
    if mode == 'red':
        val = hand * (0.42 + 0.5 * fire) + rim * fire * 1.1
        return grade_value(val, RED, 0.7)
    core = lut(COLD)[(np.clip(hand * (0.78 + 0.2 * fire), 0, 1) * 1023).astype(np.int32)]
    flame = lut(RED)[(np.clip(rim * fire * 1.15 + hand * fire * 0.25, 0, 1) * 1023).astype(np.int32)]
    out = 1 - (1 - core) * (1 - flame)                            # screen
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def grade_value(v, pal, bloom):
    L = lut(pal)
    out = L[(np.clip(v, 0, 1) * 1023).astype(np.int32)]
    hot = np.clip(v - 0.62, 0, 1)
    small = cv2.resize(hot, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    b = cv2.GaussianBlur(small, (0, 0), 9) + cv2.GaussianBlur(small, (0, 0), 30) * 0.6
    b = cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    out = out + b * np.array([0.95, 0.10, 0.18], np.float32) * bloom * 1.6
    out = (1 - np.exp(-out * 1.15)) / (1 - np.exp(-1.15))
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def reader(cid, c, start=0.0, dur=None):
    src = os.path.join(ROOT, 'src', f"{c.get('src', cid)}.mp4")
    vf = []
    if 'crop' in c:
        x, y, w, h = c['crop']
        vf.append(f'crop={w}:{h}:{x}:{y}')
    vf += [f'scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos', f'crop={W}:{H}', f'fps={FPS}']
    cmd = ['ffmpeg', '-v', 'error', '-ss', str(start)] + (['-t', str(dur)] if dur else []) + \
          ['-i', src, '-vf', ','.join(vf), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)


def preview(ids):
    out = os.environ.get('PREVIEW_DIR', '/tmp/grade_preview')
    os.makedirs(out, exist_ok=True)
    rows = []
    for cid in ids:
        c = CFG[cid]
        src = os.path.join(ROOT, 'src', f"{c.get('src', cid)}.mp4")
        dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', src],
                                   capture_output=True, text=True).stdout)
        tiles = []
        for t in (dur * 0.2, dur * 0.5, dur * 0.8):
            fr = next(reader(cid, c, t, 0.1))
            g = grade(fr, c)
            tiles += [cv2.resize(fr, (384, 216)), cv2.resize(g, (384, 216))]
        row = np.hstack(tiles)[..., ::-1].copy()
        cv2.putText(row, cid, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        rows.append(row)
    for k in range(0, len(rows), 5):
        cv2.imwrite(f'{out}/grade_{k // 5}.jpg', np.vstack(rows[k:k + 5]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print('preview →', out)


def encode(ids):
    os.makedirs(os.path.join(ROOT, 'graded'), exist_ok=True)
    for cid in ids:
        c = CFG[cid]
        dst = os.path.join(ROOT, 'graded', f'{cid}.mp4')
        enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                                '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '14',
                                '-pix_fmt', 'yuv420p', '-g', '15', dst], stdin=subprocess.PIPE)
        n = 0
        if c.get('burn'):
            fire = reader('52312', dict(CFG['52312'], src='52312'), 2.0)
            for fr, ff in zip(reader(cid, c), fire):
                enc.stdin.write(burn_hands(fr, ff, c['burn']).tobytes())
                n += 1
        else:
            for fr in reader(cid, c):
                enc.stdin.write(grade(fr, c).tobytes())
                n += 1
        enc.stdin.close()
        enc.wait()
        print(f'{cid}: {n} frames → {dst}', flush=True)


if __name__ == '__main__':
    mode, ids = sys.argv[1], sys.argv[2:] or list(CFG)
    (preview if mode == 'preview' else encode)(ids)
