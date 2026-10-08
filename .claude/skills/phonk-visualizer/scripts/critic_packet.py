#!/usr/bin/env python3
"""Build a blind review packet for the critic subagents (see references/critics.md).

Usage:
  critic_packet.py --ref REF.mp4 --cand CANDIDATE.mp4 [--cal KNOWN_BAD.mp4] --out PACKET_DIR [--seed N]

The reference is shown as REF (anchor = 10/10). Candidate and the known-bad
calibration clip are shuffled into X / Y; the mapping goes to PACKET_DIR/.key.json,
which critics must not read. For every clip the packet holds:
  sheets/       numbered contact sheets of every frame
  transitions/  12-frame strips around each detected flash / black frame / hard cut
  motion/       16 consecutive frames from the busiest stretches
  detail/       full-resolution centre crops
  hook.png      the first second;  seam.png  last | first frame
  audio.png     spectrogram + RMS with kicks (red) and visual events (cyan)
  metrics.txt   raw numbers (no thresholds, no hints which clip is which)
"""
import argparse
import json
import os
import random
import subprocess
import tempfile

import cv2
import numpy as np


def read_all(path, size=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.resize(f, size, interpolation=cv2.INTER_AREA) if size else f)
    return out, fps


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img


def grid(imgs, cols):
    blank = np.zeros_like(imgs[0])
    imgs = imgs + [blank] * ((-len(imgs)) % cols)
    return np.vstack([np.hstack(imgs[i:i + cols]) for i in range(0, len(imgs), cols)])


def events(gray):
    mean = gray.mean(axis=(1, 2))
    diff = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))
    ev = []
    for i in range(1, len(gray)):
        kind = None
        if mean[i] > 170 and mean[i - 1] <= 170:
            kind = 'flash'
        elif mean[i] < 8 and mean[i - 1] >= 8:
            kind = 'black'
        elif diff[i - 1] > 40:
            kind = 'cut'
        if kind and (not ev or i - ev[-1][0] > 8):
            ev.append((i, kind))
    return ev, mean


def metrics(gray, fps):
    n = len(gray)
    starts = [0]
    for i in range(1, n):
        if (np.abs(gray[i] - gray[starts[-1]]) > 10).mean() > 0.002:
            starts.append(i)
    ch = np.array([(np.abs(gray[b] - gray[a]) > 25).mean() for a, b in zip(starts[:-1], starts[1:])])
    mean = gray.mean(axis=(1, 2))
    return {
        'frames': n, 'fps': fps, 'unique_drawings': len(starts),
        'effective_fps': round(len(starts) / (n / fps), 1),
        'change_per_drawing_median': round(float(np.median(ch)), 3),
        'drawings_under_5pct_change': round(float((ch < 0.05).mean()), 3),
        'luma_jump_between_drawings_median': round(float(np.median(np.abs(np.diff(mean[starts])))), 1),
        'luma_p5': round(float(np.percentile(gray, 5)), 1),
        'share_darker_than_12': round(float((gray < 12).mean()), 3),
        'loop_seam_diff': round(float(np.abs(gray[-1] - gray[0]).mean()), 1),
    }


def pink_share(frames):
    px = np.concatenate([cv2.cvtColor(f, cv2.COLOR_BGR2HSV).reshape(-1, 3) for f in frames[::3]]).astype(np.int32)
    h, s, v = px[:, 0] * 2, px[:, 1], px[:, 2]
    reds = (s > 90) & (v > 70)
    crimson = reds & ((h >= 345) | (h <= 12))
    pink = reds & (h >= 300) & (h < 345)
    return round(float(pink.sum() / max(1, pink.sum() + crimson.sum())), 3)


def audio_plot(path, vis_events, fps, out_png):
    try:
        import librosa
        import librosa.display
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return 'librosa/matplotlib missing'
    wav = os.path.join(tempfile.mkdtemp(), 'a.wav')
    r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', path, '-vn', '-ac', '1', '-ar', '22050', wav])
    if r.returncode or not os.path.exists(wav):
        return 'no audio track'
    y, sr = librosa.load(wav, sr=22050)
    onset = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    fig, ax = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=256)), ref=np.max)
    librosa.display.specshow(S, sr=sr, hop_length=256, x_axis='time', y_axis='log', ax=ax[0], cmap='magma')
    rms = librosa.feature.rms(y=y, hop_length=256)[0]
    ax[1].plot(librosa.times_like(rms, sr=sr, hop_length=256), rms, color='k')
    for t in onset:
        ax[1].axvline(t, color='red', lw=0.6)
    for f, kind in vis_events:
        ax[1].axvline(f / fps, color='cyan', lw=2)
        ax[1].text(f / fps, rms.max(), kind, rotation=90, fontsize=8)
    ax[1].set_title('RMS; red = audio onsets, cyan = visual events (flash / black / cut)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=55)
    plt.close()
    return f'audio onsets (s): {np.round(onset, 2).tolist()}'


def build(path, out):
    for d in ('sheets', 'transitions', 'motion', 'detail'):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', path, '-vf',
                    "drawtext=text='%{n}':x=6:y=6:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.6,"
                    'scale=384:-2,tile=5x6', '-vsync', '0', os.path.join(out, 'sheets', 'sheet_%02d.png')], check=True)
    full, fps = read_all(path)
    small = [cv2.resize(f, (480, 270), interpolation=cv2.INTER_AREA) for f in full]
    gray = np.array([cv2.cvtColor(cv2.resize(f, (320, 180)), cv2.COLOR_BGR2GRAY) for f in full]).astype(np.float32)
    ev, _ = events(gray)
    for k, (f, kind) in enumerate(ev[:8]):
        idx = [i for i in range(f - 4, f + 8) if 0 <= i < len(small)]
        cv2.imwrite(os.path.join(out, 'transitions', f't{k:02d}_{kind}_f{f}.jpg'),
                    grid([label(small[i], str(i)) for i in idx], 4), [cv2.IMWRITE_JPEG_QUALITY, 88])
    motion = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))
    win = np.convolve(motion, np.ones(16), 'valid')
    picks = []
    for i in np.argsort(-win):
        if all(abs(i - p) > 24 for p in picks):
            picks.append(int(i))
        if len(picks) == 3:
            break
    for k, s in enumerate(sorted(picks)):
        cv2.imwrite(os.path.join(out, 'motion', f'm{k}_f{s}-{s + 15}.jpg'),
                    grid([label(small[i], str(i)) for i in range(s, min(s + 16, len(small)))], 4),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    n = len(full)
    for k, f in enumerate([n // 6, n // 2, (5 * n) // 6]):
        h, w = full[f].shape[:2]
        crop = full[f][h // 4:h // 4 + h // 2, w // 4:w // 4 + w // 2]
        cv2.imwrite(os.path.join(out, 'detail', f'd{k}_f{f}.png'), crop)
    cv2.imwrite(os.path.join(out, 'hook.jpg'), grid([label(small[i], str(i)) for i in range(0, min(30, n), 2)], 5),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(os.path.join(out, 'seam.jpg'), np.hstack([label(small[-1], 'last'), label(small[0], 'first')]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    m = metrics(gray, fps)
    m['pink_among_reds'] = pink_share(full)
    m['visual_events'] = [f'{f}:{k}' for f, k in ev]
    m['audio'] = audio_plot(path, ev, fps, os.path.join(out, 'audio.png'))
    with open(os.path.join(out, 'metrics.txt'), 'w') as fh:
        for k, v in m.items():
            fh.write(f'{k}: {v}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--cand', required=True)
    ap.add_argument('--cal', default=None, help='known-bad clip used to check critic calibration (recommended)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    labels = ['X', 'Y']
    rng.shuffle(labels)
    clips = [('REF', a.ref), (labels[0], a.cand)] + ([(labels[1], a.cal)] if a.cal else [])
    key = {lab: os.path.abspath(p) for lab, p in clips}
    key.update({'candidate': labels[0], 'calibration': labels[1] if a.cal else None})
    json.dump(key, open(os.path.join(a.out, '.key.json'), 'w'), indent=1)
    for lab, path in clips:
        build(path, os.path.join(a.out, lab))
        print('built', lab)
    with open(os.path.join(a.out, 'README.md'), 'w') as fh:
        others = ', '.join(lab for lab, _ in clips[1:])
        fh.write(f'# Review packet\n\nREF = reference, scored 10/10 by the client. Clips to score: {others} '
                 '(labels are random). Each folder: sheets/, transitions/, motion/, detail/, hook.jpg, seam.jpg, '
                 'audio.png, metrics.txt. Do not open .key.json.\n')
    print('packet ready:', a.out)


if __name__ == '__main__':
    main()
