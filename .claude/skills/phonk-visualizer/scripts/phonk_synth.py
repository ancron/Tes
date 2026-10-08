#!/usr/bin/env python3
"""phonk_synth — sample-free phonk instruments + a tiny arranger (numpy/scipy only).

Use as a library:
    from phonk_synth import *
    m = Mixer(seconds=12.6)
    m.add(kick(), at=0.0, gain=0.9)
    m.add(bass808([(49.0, 1.5)]), at=0.0, gain=0.8, sidechain=True)
    m.kicks.append(0.0)
    m.render('track.wav')

Or run directly for a demo: phonk_synth.py out.wav
Instruments: kick, bass808 (glides + distortion), clap, hat, cowbell, pad, riser, reverse_swell, impact.
Mixer: stereo bus with pan, convolution reverb send, kick sidechain on flagged tracks, vinyl, limiter.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import scipy.signal as ss
from scipy.io import wavfile

SR = 44100
A4 = 440.0
NOTE = {n: i for i, n in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])}
NOTE.update({'Db': 1, 'Eb': 3, 'Gb': 6, 'Ab': 8, 'Bb': 10})


def hz(name: str) -> float:
    """'G1', 'Bb4', 'C#5' → frequency."""
    p, o = (name[:-1], int(name[-1]))
    return A4 * 2 ** ((NOTE[p] + 12 * (o + 1) - 69) / 12)


def t_axis(sec):
    return np.arange(int(sec * SR)) / SR


def lp(x, fc, order=2):
    b, a = ss.butter(order, min(fc, SR / 2 - 100) / (SR / 2), 'low')
    return ss.lfilter(b, a, x)


def hp(x, fc, order=2):
    b, a = ss.butter(order, fc / (SR / 2), 'high')
    return ss.lfilter(b, a, x)


def bp(x, lo, hi, order=2):
    b, a = ss.butter(order, [lo / (SR / 2), min(hi, SR / 2 - 100) / (SR / 2)], 'band')
    return ss.lfilter(b, a, x)


def drive(x, amt):
    return np.tanh(x * amt) / np.tanh(amt)


# ------------------------------------------------------------------ instruments

def kick(f_hi=180.0, f_lo=44.0, sweep=0.035, decay=0.42, click=0.25, amt=2.2):
    t = t_axis(decay * 3)
    f = f_lo + (f_hi - f_lo) * np.exp(-t / sweep)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / decay)
    cl = hp(np.random.default_rng(1).normal(0, 1, len(t)), 2500) * np.exp(-t / 0.003) * click
    return drive(body + cl, amt) * 0.95


def bass808(notes, glide=0.06, drive_amt=4.0, tone=2600.0, punch=0.9, release=0.06):
    """notes: [(freq_hz, dur_s), ...] played legato; frequency glides `glide` s into each new note."""
    total = sum(d for _, d in notes)
    t = t_axis(total)
    f = np.zeros_like(t)
    env = np.ones_like(t)
    pos = 0
    prev = notes[0][0]
    for i, (fr, d) in enumerate(notes):
        n = int(d * SR)
        tt = np.arange(n) / SR
        target = fr * (1 + punch * np.exp(-tt / 0.018))          # pitch "kick" at onset
        if i > 0 and abs(fr - prev) > 0.5:
            g = np.clip(tt / glide, 0, 1)
            target = prev + (target - prev) * (g * g * (3 - 2 * g))
        f[pos:pos + n] = target
        pos += n
        prev = fr
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) + 0.18 * np.sin(2 * ph) + 0.05 * np.sin(3 * ph)
    a = int(0.003 * SR)
    env[:a] = np.linspace(0, 1, a)
    r = int(release * SR)
    env[-r:] *= np.linspace(1, 0, r)
    y = drive(x * env, drive_amt)
    return lp(y, tone) * 0.9


def clap(decay=0.16, tone=(800, 2600)):
    t = t_axis(0.6)
    rng = np.random.default_rng(7)
    n = rng.normal(0, 1, len(t))
    env = np.zeros_like(t)
    for k, off in enumerate((0.0, 0.011, 0.022)):
        m = t >= off
        env[m] += np.exp(-(t[m] - off) / (0.006 if k < 2 else decay))
    body = np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.05) * 0.4
    return drive(bp(n, *tone) * env * 1.6 + body, 1.6) * 0.8


def hat(open_=False):
    t = t_axis(0.35 if open_ else 0.08)
    n = np.random.default_rng(3 if open_ else 4).normal(0, 1, len(t))
    return hp(n, 7000, 4) * np.exp(-t / (0.11 if open_ else 0.022)) * 0.5


def cowbell(freq, decay=0.09, ratio=1.48, amt=1.6):
    """TR-808-style cowbell tuned to `freq`: two square oscillators through a band-pass."""
    t = t_axis(decay * 5)
    sq = np.sign(np.sin(2 * np.pi * freq * t)) + np.sign(np.sin(2 * np.pi * freq * ratio * t)) * 0.8
    sq = lp(sq, 9000)
    env = np.exp(-t / decay) * 0.85 + np.exp(-t / 0.012) * 0.4
    return drive(bp(sq, freq * 0.8, freq * 4.5) * env, amt) * 0.55


def pad(freqs, sec, cutoff=1100.0, attack=0.6, detune=0.008):
    t = t_axis(sec)
    x = np.zeros_like(t)
    for f in freqs:
        for d in (-detune, 0, detune):
            ph = (t * f * (1 + d) + np.random.default_rng(int(f)).random()) % 1.0
            x += 2 * ph - 1                                   # naive saw, tamed by the LP below
    x = lp(x / (len(freqs) * 3), cutoff, 4)
    env = np.minimum(1, t / attack) * np.minimum(1, (sec - t) / 0.4)
    return x * env * 0.6


def riser(sec, lo=300.0, hi=9000.0):
    t = t_axis(sec)
    n = np.random.default_rng(11).normal(0, 1, len(t))
    out = np.zeros_like(t)
    seg = int(0.02 * SR)
    for i in range(0, len(t), seg):                           # stepped band-pass sweep
        k = i / len(t)
        fc = lo * (hi / lo) ** k
        out[i:i + seg] = bp(n[i:i + seg + 2000], fc * 0.7, fc * 1.4)[:len(out[i:i + seg])]
    return out * (t / sec) ** 2 * 0.6


def reverse_swell(sec=0.4):
    t = t_axis(sec)
    n = hp(np.random.default_rng(5).normal(0, 1, len(t)), 3000)
    return n * (t / sec) ** 3 * 0.5


def impact(sec=2.0):
    t = t_axis(sec)
    boom = np.sin(2 * np.pi * (38 + 60 * np.exp(-t / 0.08)) * t) * np.exp(-t / 0.7)
    crash = hp(np.random.default_rng(9).normal(0, 1, len(t)), 1500) * np.exp(-t / 0.5) * 0.35
    return drive(boom + crash, 1.8) * 0.9


# ------------------------------------------------------------------ mixer

def reverb_ir(sec=2.2, seed=0, damp=6000):
    t = t_axis(sec)
    rng = np.random.default_rng(seed)
    L = lp(rng.normal(0, 1, len(t)), damp) * np.exp(-t / (sec / 5.5))
    R = lp(rng.normal(0, 1, len(t)), damp) * np.exp(-t / (sec / 5.5))
    return np.stack([L, R], 1) / np.sqrt(np.sum(L ** 2))


class Mixer:
    def __init__(self, seconds):
        self.n = int(seconds * SR)
        self.dry = np.zeros((self.n, 2))
        self.sc = np.zeros((self.n, 2))      # sidechained bus
        self.send = np.zeros(self.n)         # reverb send (mono in)
        self.kicks: list[float] = []         # kick times for the sidechain

    def add(self, buf, at, gain=1.0, pan=0.0, rev=0.0, sidechain=False):
        s = int(at * SR)
        if s >= self.n:
            return
        b = np.asarray(buf, np.float64)[: self.n - s] * gain
        l, r = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        bus = self.sc if sidechain else self.dry
        bus[s:s + len(b), 0] += b * l * 1.414
        bus[s:s + len(b), 1] += b * r * 1.414
        self.send[s:s + len(b)] += b * rev

    def render(self, path, vinyl=0.006, sc_depth=0.75, sc_release=0.09, ceiling=0.89):
        t = np.arange(self.n) / SR
        g = np.ones(self.n)
        for k in self.kicks:
            m = t >= k
            g[m] = np.minimum(g[m], 1 - sc_depth * np.exp(-(t[m] - k) / sc_release))
        mix = self.dry + self.sc * g[:, None]
        ir = reverb_ir()
        wet = np.stack([ss.fftconvolve(self.send, ir[:, c])[: self.n] for c in range(2)], 1)
        mix += wet * 0.9
        if vinyl:
            rng = np.random.default_rng(21)
            crackle = (rng.random(self.n) < 0.0007) * rng.normal(0, 1, self.n) * 0.5
            hiss = lp(rng.normal(0, 1, self.n), 5000) * 0.25
            mix += ((crackle + hiss) * vinyl)[:, None]
        mix = drive(mix * 1.15, 1.3)                      # glue saturation
        mix = mix / (np.max(np.abs(mix)) + 1e-9) * ceiling
        wavfile.write(path, SR, (mix * 32767).astype(np.int16))
        return path


def demo(path):
    bpm, bar = 152, 4 * 60 / 152
    m = Mixer(4 * bar)
    for b in range(4):
        k = b * bar
        m.add(kick(), k, 0.9); m.kicks.append(k)
        m.add(clap(), k + bar / 2, 0.5, rev=0.25)
        for s in range(8):
            m.add(hat(), k + s * bar / 8, 0.18, pan=0.3)
        for s, nt in zip((0, 3, 6, 8, 10, 12, 14), ('G4', 'G4', 'Bb4', 'G4', 'F4', 'D4', 'F4')):
            m.add(cowbell(hz(nt)), k + s * bar / 16, 0.35, pan=-0.2, rev=0.3)
    m.add(bass808([(hz('G1'), bar * 2), (hz('Eb1'), bar), (hz('F1'), bar)]), 0, 0.8, sidechain=True)
    m.render(path)
    print("wrote", path)


if __name__ == "__main__":
    demo(sys.argv[1] if len(sys.argv) > 1 else "phonk_demo.wav")
