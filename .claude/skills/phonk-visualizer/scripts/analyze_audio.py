#!/usr/bin/env python3
"""Music breakdown for beat-synced edits.

Usage: analyze_audio.py MEDIA OUT_DIR [--fps 30]

Prints kick times (and frame numbers), tempo, bass note track, key estimate,
energy dips (breaks) and the drop; writes OUT_DIR/spectrum.png
(log spectrogram, CQT notes, chroma, RMS) to look at.
Needs: librosa, matplotlib, scipy (pip install librosa matplotlib scipy).
"""
import argparse, os, subprocess, tempfile
import numpy as np
import librosa, librosa.display
import scipy.signal as ss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("media"); ap.add_argument("out"); ap.add_argument("--fps", type=float, default=30)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    wav = os.path.join(tempfile.mkdtemp(), "a.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.media, "-vn", "-ac", "1", "-ar", "44100", wav], check=True)
    y, sr = librosa.load(wav, sr=44100)
    dur = len(y) / sr

    # kicks: rising energy in the kick-sweep band
    hop = 128
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
    band = S[(f > 150) & (f < 450)].sum(0)  # kick pitch-sweep band
    d = ss.medfilt(np.maximum(0, np.diff(band, prepend=band[0])), 3)
    pk, _ = ss.find_peaks(d, height=d.max() * 0.25, distance=int(0.25 * sr / hop))
    kicks = t[pk]
    iv = np.diff(kicks)
    period = float(np.median(iv)) if len(iv) else 0

    # key
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(1)
    scores = [(np.corrcoef(np.roll(MAJ, i), chroma)[0, 1], NAMES[i] + " major") for i in range(12)]
    scores += [(np.corrcoef(np.roll(MIN, i), chroma)[0, 1], NAMES[i] + " minor") for i in range(12)]
    key = max(scores)[1]

    # bass notes
    f0, _, _ = librosa.pyin(y, fmin=30, fmax=220, sr=sr, frame_length=8192, hop_length=1024)
    tt = librosa.times_like(f0, sr=sr, hop_length=1024)
    notes, prev = [], None
    for ti, x in zip(tt, f0):
        nm = librosa.hz_to_note(x) if not np.isnan(x) else '-'
        if nm != prev:
            notes.append(f"{ti:.2f}:{nm}")
            prev = nm

    # energy: breaks and drop
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    rt = librosa.times_like(rms, sr=sr, hop_length=512)
    sm = np.convolve(rms, np.ones(9) / 9, mode="same")
    low = sm < 0.5 * np.median(sm)
    breaks, i = [], 0
    while i < len(low):
        if low[i]:
            j = i
            while j < len(low) and low[j]:
                j += 1
            if rt[min(j, len(rt) - 1)] - rt[i] > 0.15:
                breaks.append((round(float(rt[i]), 2), round(float(rt[min(j, len(rt) - 1)]), 2)))
            i = j
        else:
            i += 1
    drop = None
    if breaks:
        after = kicks[kicks > breaks[0][1] - 0.1]
        drop = float(after[0]) if len(after) else None

    print(f"duration {dur:.2f}s")
    print(f"kick period {period:.3f}s -> {60 / period:.1f} BPM (x2 = {120 / period:.1f} half-time)" if period else "no kicks")
    print("kicks (s):", " ".join(f"{k:.3f}" for k in kicks))
    print(f"kicks (frames @{a.fps:g}):", " ".join(str(round(k * a.fps)) for k in kicks))
    print("key estimate:", key)
    print("bass notes:", " ".join(notes))
    print("breaks (low energy):", breaks)
    print("drop:", f"{drop:.3f}s = frame {round(drop * a.fps)}" if drop else None)

    fig, ax = plt.subplots(4, 1, figsize=(20, 16), sharex=True)
    Sd = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=4096, hop_length=256)), ref=np.max)
    librosa.display.specshow(Sd, sr=sr, hop_length=256, x_axis='time', y_axis='log', ax=ax[0], cmap='magma')
    ax[0].set_title('log spectrogram')
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=256, n_bins=84, fmin=librosa.note_to_hz('C1')))
    librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max), sr=sr, hop_length=256, x_axis='time',
                             y_axis='cqt_note', ax=ax[1], cmap='magma')
    ax[1].set_title('CQT (notes)')
    ch = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=256)
    librosa.display.specshow(ch, sr=sr, hop_length=256, x_axis='time', y_axis='chroma', ax=ax[2])
    ax[2].set_title('chroma')
    ax[3].plot(rt, rms)
    for k in kicks:
        ax[3].axvline(k, color='red', lw=.6)
    ax[3].set_title('RMS + detected kicks')
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "spectrum.png"), dpi=60)
    print("plot:", os.path.join(a.out, "spectrum.png"))


if __name__ == "__main__":
    main()
