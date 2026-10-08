#!/usr/bin/env python3
"""Original phonk track for "Burning Rose": 152 BPM half-time, G minor, 8 bars, seamless loop.

Writes OUT/track.wav and OUT/events.json (kick/clap times, bars, break, drop) for the video.
"""
import json
import os
import sys

import numpy as np
import scipy.signal as ss
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../.claude/skills/phonk-visualizer/scripts'))
from phonk_synth import (SR, Mixer, bass808, clap, cowbell, drive, hat, hz, impact, kick, lp, pad,  # noqa: E402
                         reverse_swell, riser)

OUT = sys.argv[1] if len(sys.argv) > 1 else '.'
BPM = 152
BEAT = 60 / BPM
BAR = 4 * BEAT
STEP = BAR / 16
BARS = 8
LENGTH = BARS * BAR
TAIL = 2.5  # rendered past the end and folded onto the start → seamless loop


def at(bar, step=0.0):
    return bar * BAR + step * STEP


def sweep_lp(x, f0, f1, block=1024):
    """Time-varying low-pass (filter sweep) with carried filter state."""
    y = np.zeros_like(x)
    zi = None
    nb = int(np.ceil(len(x) / block))
    for i in range(nb):
        fc = f0 * (f1 / f0) ** (i / max(1, nb - 1))
        b, a = ss.butter(2, fc / (SR / 2), 'low')
        if zi is None:
            zi = ss.lfilter_zi(b, a) * x[0]
        seg = x[i * block:(i + 1) * block]
        y[i * block:(i + 1) * block], zi = ss.lfilter(b, a, seg, zi=zi)
    return y


m = Mixer(LENGTH + TAIL)
ev = {'bpm': BPM, 'bar_s': BAR, 'beat_s': BEAT, 'length_s': LENGTH, 'kicks': [], 'claps': [],
      'break_s': at(2, 12), 'drop_s': at(3)}


def K(t, gain=0.9, strength=1.0, sc=True):
    m.add(kick(), t, gain)
    if sc:
        m.kicks.append(t)
    if t < LENGTH:
        ev['kicks'].append([round(t, 4), strength])


def C(t, gain=0.5):
    m.add(clap(), t, gain, rev=0.28)
    ev['claps'].append(round(t, 4))


COW_A = [(0, 'G4'), (3, 'G4'), (6, 'Bb4'), (8, 'G4'), (10, 'F4'), (12, 'D4'), (14, 'F4')]
COW_B = [(0, 'G4'), (3, 'G4'), (6, 'Bb4'), (8, 'C5'), (10, 'Bb4'), (12, 'G4'), (14, 'F4'), (15, 'D5')]


def cow_bar(bar, pattern, gain=0.33, far=False):
    for s, n in pattern:
        x = cowbell(hz(n))
        if far:
            x = lp(x, 1400)
        m.add(x, at(bar, s), gain, pan=-0.25 if s % 2 else 0.2, rev=0.6 if far else 0.3)


# ---- bars 1–2 (0,1): intro — smouldering
for b in (0, 1):
    K(at(b, 0), 0.3, 0.45)
    K(at(b, 8), 0.24, 0.35)
intro_bass = bass808([(hz('G1'), 2 * BAR)], drive_amt=3.0)
m.add(sweep_lp(intro_bass, 520, 1300), at(0), 0.4, sidechain=True)
m.add(pad([hz('G3'), hz('Bb3'), hz('D4')], 2 * BAR + 0.5), at(0), 0.22, rev=0.55)
cow_bar(1, COW_A, 0.2, far=True)

# ---- bar 3 (2): ignition / build, last beat = break
K(at(2, 0), 0.8, 0.7)
K(at(2, 8), 0.75, 0.6)
C(at(2, 8), 0.45)
for s in range(0, 12):
    if s % 2 == 0 or s >= 8:
        m.add(hat(), at(2, s), 0.15 + 0.01 * s, pan=0.3)
cow_bar(2, COW_A[:5], 0.28)
m.add(bass808([(hz('Eb1'), 2 * BEAT), (hz('F1'), BEAT)]), at(2), 0.8, sidechain=True)
m.add(riser(3 * BEAT + 0.05), at(2), 0.55)
m.add(reverse_swell(BEAT * 0.95), at(2, 12) + BEAT * 0.05, 0.6)

# ---- bars 4–7 (3..6): drop
m.add(impact(), at(3), 0.8, rev=0.35)
for b in range(3, 7):
    K(at(b, 0), 0.95, 1.0)
    K(at(b, 10), 0.85, 0.75)
    if b in (4, 6):
        K(at(b, 14), 0.8, 0.6)
    C(at(b, 8), 0.55)
    for s in range(0, 16, 2):
        m.add(hat(), at(b, s), 0.16, pan=0.3)
    if b in (4, 6):
        for s in (12, 13, 14, 15):
            m.add(hat(), at(b, s), 0.13, pan=-0.3)
        for k in range(3):  # triplet roll into the next bar
            m.add(hat(), at(b, 15) + k * STEP / 3, 0.1, pan=-0.3)
    m.add(hat(open_=True), at(b, 6), 0.12, pan=0.35)
    cow_bar(b, COW_A if b % 2 == 1 else COW_B)
drop_bass = bass808([(hz('G1'), BAR), (hz('G1'), 2 * BEAT), (hz('Bb1'), 2 * BEAT), (hz('Eb1'), BAR),
                     (hz('D1'), 2 * BEAT), (hz('F1'), 2 * BEAT)], drive_amt=4.5)
m.add(drop_bass, at(3), 0.85, sidechain=True)
for b, ch in zip(range(3, 7), (['G3', 'Bb3', 'D4'], ['G3', 'Bb3', 'D4'], ['Eb3', 'G3', 'Bb3'], ['D3', 'F3', 'A3'])):
    m.add(pad([hz(n) for n in ch], BAR + 0.3), at(b), 0.14, rev=0.4, sidechain=True)

# ---- bar 8 (7): afterglow — back to the intro state (loop seam)
K(at(7, 0), 0.4, 0.5)
K(at(7, 8), 0.26, 0.35)
outro_bass = bass808([(hz('G1'), BAR + 0.03)], drive_amt=3.0)
m.add(sweep_lp(outro_bass, 1800, 520), at(7), 0.4, sidechain=True)
m.add(pad([hz('G3'), hz('Bb3'), hz('D4')], BAR + 0.6), at(7), 0.22, rev=0.55)
cow_bar(7, COW_B[:4], 0.18, far=True)
# what the loop's first bar plays, continued past the end so tails fold correctly
K(at(8, 0), 0.3, 0.45)

os.makedirs(OUT, exist_ok=True)
tmp = os.path.join(OUT, '_full.wav')
m.render(tmp)
sr, x = wavfile.read(tmp)
x = x.astype(np.float32) / 32767
n = int(LENGTH * SR)
loop = x[:n].copy()
tail = x[n:]
fade = np.linspace(1, 0, len(tail))[:, None] ** 0.5
loop[:len(tail)] += tail * fade * 0.85   # wrap reverb/808 tails onto the start
loop = drive(loop * 1.05, 1.2)
loop = loop / np.max(np.abs(loop)) * 0.89
wavfile.write(os.path.join(OUT, 'track.wav'), SR, (loop * 32767).astype(np.int16))
os.remove(tmp)
json.dump(ev, open(os.path.join(OUT, 'events.json'), 'w'), indent=1)
print(f"track {LENGTH:.3f}s, {len(ev['kicks'])} kicks, drop at {ev['drop_s']:.3f}s → {OUT}/track.wav")
