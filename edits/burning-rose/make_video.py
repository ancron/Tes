#!/usr/bin/env python3
"""Burning Rose — dark phonk visualizer built with fxkit (see PLAN.md for the shot table).

Usage: make_video.py OUT_DIR [--preview N] [--only f1,f2,...]
  OUT_DIR must contain track.wav + events.json from make_track.py.
  --preview N : render at 1/N resolution (fast look-dev)
  --only      : render just these frames to PNG (look-dev), no video
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../.claude/skills/phonk-visualizer/scripts'))
from fxkit import (DUST, PETAL, SHARD, SPARK, FFmpegWriter, HoldSchedule, Noise, Palette, Particles,  # noqa: E402
                   camera, composite_particles, finish, gblur, glow, gradient_map, light_ring, luma, smoothstep,
                   soft_disc, toon_fire, zoom_blur)

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--preview', type=int, default=1)
ap.add_argument('--only', default='')
A = ap.parse_args()

EV = json.load(open(os.path.join(A.out, 'events.json')))
FPS = 30
N = int(round(EV['length_s'] * FPS))            # 379
DIV = A.preview
W, H = 1920 // DIV, 1080 // DIV
S = 1.0 / DIV                                    # pixel scale
CX, CY = W / 2, H * 0.50
R = 300 * S                                      # rose radius
SQ = 0.80                                        # 3/4 top view squash

F_BUILD, F_BREAK, F_DROP, F_ZOOM, F_FLASH, F_REBIRTH_END, F_OUTRO = 95, 130, 142, 237, 284, 320, 332
KICKS = [(int(round(t * FPS)), s) for t, s in EV['kicks']]
CLAPS = [int(round(t * FPS)) for t in EV['claps']]

HOLDS = HoldSchedule([(0, F_BUILD, 3), (F_BUILD, F_BREAK, 2), (F_BREAK, F_DROP - 1, 3), (F_DROP - 1, F_DROP, 1),
                      (F_DROP, F_DROP + 8, 1), (F_DROP + 8, F_ZOOM, 2), (F_ZOOM, 270, 2), (270, F_FLASH, 1),
                      (F_FLASH, F_FLASH + 4, 1), (F_FLASH + 4, F_REBIRTH_END, 2), (F_REBIRTH_END, F_REBIRTH_END + 2, 1),
                      (F_REBIRTH_END + 2, F_OUTRO, 2), (F_OUTRO, N, 3)], N)


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def ease_io(x):
    x = min(max(x, 0.0), 1.0)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def env(f, hits, decay):
    v = 0.0
    for hf, s in hits:
        if 0 <= f - hf < decay * 6:
            v = max(v, s * math.exp(-(f - hf) / decay))
    return v


def ramp(f, a, b):
    return min(max((f - a) / (b - a), 0.0), 1.0)


# ------------------------------------------------------------------ the rose

class Rose:
    RINGS = [(5, 0.30, 0.04, 1.30), (6, 0.52, 0.12, 1.25), (7, 0.74, 0.24, 1.20), (8, 0.97, 0.40, 1.12)]

    def __init__(self, seed=4):
        rng = np.random.default_rng(seed)
        self.petals = []
        for ri, (n, ro, rin, wide) in enumerate(self.RINGS):
            for k in range(n):
                a = 2 * np.pi * k / n + ri * 0.55 + rng.uniform(-0.12, 0.12)
                self.petals.append(dict(ring=ri, ang=a, ro=ro * rng.uniform(0.94, 1.04), ri_=rin,
                                        half=np.pi / n * wide, notch=rng.uniform(0.03, 0.08)))
        self.n = len(self.petals)
        self.off = np.zeros((self.n, 2), np.float32)
        self.vel = np.zeros((self.n, 2), np.float32)
        self.rot = np.zeros(self.n, np.float32)
        self.rotv = np.zeros(self.n, np.float32)
        self.scl = np.ones(self.n, np.float32)
        self.orbit_r = rng.uniform(330, 720, self.n).astype(np.float32) * S
        self.rng = rng

    def outline(self, i, open_=1.0):
        """Petal polygon (rose-local coords, unsquashed). Returns (poly, lip_polyline)."""
        p = self.petals[i]
        a, hw = p['ang'], p['half']
        ro, rin = p['ro'] * R * (0.92 + 0.08 * open_), p['ri_'] * R
        u = np.linspace(-1, 1, 23)
        ang_o = a + u * hw
        rad_o = ro * (1 - 0.13 * u ** 2) * (1 - p['notch'] * np.exp(-(u / 0.22) ** 2))
        lip = np.stack([np.cos(ang_o) * rad_o, np.sin(ang_o) * rad_o], 1)
        ub = np.linspace(1, -1, 9)
        ang_b = a + ub * hw * 0.5
        base = np.stack([np.cos(ang_b) * rin, np.sin(ang_b) * rin], 1)
        return np.concatenate([lip, base]), lip

    def body(self, i, frac=0.72, narrow=0.78):
        p = self.petals[i]
        a, hw = p['ang'], p['half'] * narrow
        rin = p['ri_'] * R
        ro = rin + (p['ro'] * R - rin) * frac
        u = np.linspace(-1, 1, 17)
        rad_o = ro * (1 - 0.18 * u ** 2)
        lip = np.stack([np.cos(a + u * hw) * rad_o, np.sin(a + u * hw) * rad_o], 1)
        ub = np.linspace(1, -1, 7)
        base = np.stack([np.cos(a + ub * hw * 0.45) * rin, np.sin(a + ub * hw * 0.45) * rin], 1)
        return np.concatenate([lip, base])

    def to_screen(self, i, pts, center):
        """Apply petal transform (rotation about its centroid, scale, offset), squash, translate."""
        c = pts.mean(0)
        cr, sr = math.cos(self.rot[i]), math.sin(self.rot[i])
        q = (pts - c) @ np.array([[cr, sr], [-sr, cr]], np.float32) * self.scl[i] + c
        q[:, 1] *= SQ
        return q + np.asarray(center, np.float32) + self.off[i]

    def draw(self, img, center, lit, rim, open_=1.0, alpha=1.0, fuel=None, only_rings=None, bud=1.0):
        """lit: 0 = silhouette (dark fills) → 1 = full crimson; rim: lip-light intensity."""
        base_col = Palette.VOID * (1 - lit) + Palette.WINE * lit
        body_col = Palette.VOID * 0.6 * (1 - lit) + Palette.BLOOD * lit
        core_col = Palette.WINE * (1 - lit) + Palette.CRIMSON * lit
        lip_col = Palette.CRIMSON * (1 - 0.45 * lit) + Palette.PINK * 0.45 * lit
        for i in sorted(range(self.n), key=lambda j: -self.petals[j]['ring']):   # back (outer) first
            if only_rings is not None and self.petals[i]['ring'] not in only_rings:
                continue
            poly, lip = self.outline(i, open_)
            P = self.to_screen(i, poly, center)
            B = self.to_screen(i, self.body(i), center)
            L = self.to_screen(i, lip, center)
            k = 4
            cv2.fillPoly(img, [np.round(P * k).astype(np.int32)], tuple(map(float, base_col * alpha)), cv2.LINE_AA, k.bit_length() - 1)
            cv2.fillPoly(img, [np.round(B * k).astype(np.int32)], tuple(map(float, body_col * alpha)), cv2.LINE_AA, k.bit_length() - 1)
            inner = self.to_screen(i, self.body(i, 0.38, 0.55), center)
            cv2.fillPoly(img, [np.round(inner * k).astype(np.int32)], tuple(map(float, core_col * alpha * 0.9)), cv2.LINE_AA, k.bit_length() - 1)
            th = max(1, int(round((3.2 if self.petals[i]['ring'] >= 2 else 2.4) * S * (1 + rim))))
            top = 0.45 + 0.75 * max(0.0, -math.sin(self.petals[i]['ang'] + self.rot[i])) + 0.15 * (3 - self.petals[i]['ring']) / 3
            cv2.polylines(img, [np.round(L * k).astype(np.int32)], False, tuple(map(float, lip_col * rim * alpha * top)), th,
                          cv2.LINE_AA, k.bit_length() - 1)
            if fuel is not None and self.petals[i]['ring'] >= 2:
                cv2.polylines(fuel, [np.round(L * k / 2).astype(np.int32)], False, 1.0, max(2, int(9 * S)),
                              cv2.LINE_AA, k.bit_length() - 1)
        if bud <= 0.01:
            return
        # bud: tight spiral
        t = np.linspace(0, 2.6 * 2 * np.pi, 160)
        rr = R * (0.025 + 0.13 * t / t[-1])
        sp = np.stack([np.cos(t + 0.8) * rr, np.sin(t + 0.8) * rr * SQ], 1) + np.asarray(center, np.float32)
        cv2.circle(img, (int(center[0] * 4), int(center[1] * 4)), int(R * 0.15 * 4 * bud), tuple(map(float, body_col * alpha)),
                   -1, cv2.LINE_AA, 2)
        sp = (sp - np.asarray(center, np.float32)) * bud + np.asarray(center, np.float32)
        cv2.polylines(img, [np.round(sp * 4).astype(np.int32)], False, tuple(map(float, lip_col * rim * 0.9 * alpha * bud)),
                      max(1, int(2.6 * S * (1 + rim))), cv2.LINE_AA, 2)

    def explode(self, center):
        for i, p in enumerate(self.petals):
            d = np.array([math.cos(p['ang']), math.sin(p['ang']) * SQ])
            sp = self.rng.uniform(700, 1500) * S * (1.25 - 0.15 * p['ring'])
            self.vel[i] = d * sp + self.rng.normal(0, 120 * S, 2)
            self.rotv[i] = self.rng.uniform(-7, 7)

    def step_vortex(self, dt, center, strength=1.0):
        home = np.array([[math.cos(p['ang']) * p['ro'] * R * 0.6, math.sin(p['ang']) * p['ro'] * R * 0.6 * SQ]
                         for p in self.petals], np.float32)
        pos = home + self.off
        d = pos - 0
        dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-3
        tang = np.concatenate([-d[:, 1:2], d[:, 0:1]], 1) / dist
        radial = -(dist - self.orbit_r[:, None]) * 2.2 * d / dist
        self.vel *= math.exp(-1.4 * dt)
        self.vel += (tang * 520 * S * strength + radial) * dt
        self.off += self.vel * dt
        self.rot += self.rotv * dt


# ------------------------------------------------------------------ scene state & simulation

nz = Noise(11)
fire_nz = Noise(23)
rose = Rose()
P = Particles(seed=5)          # bursts / vortex shards
E = Particles(seed=9)          # embers
home_off = None
spiral_start = None
rebirth_rng = np.random.default_rng(77)


def simulate(f):
    """Advance simulations by one frame (called for every frame, rendered or held)."""
    global spiral_start
    dt = 1 / FPS
    c = (CX, CY)
    # embers: always a few, many in rebirth
    rate = 3 if f < F_DROP else (1 if f < F_FLASH else (10 if f < F_OUTRO else 3))
    if F_BREAK <= f < F_DROP:
        rate = 0
    E.emit(rate, (CX, CY + 40 * S), speed=(30 * S, 150 * S), spread=(-np.pi * 0.85, -np.pi * 0.15),
           radius=R * (1.0 if (f < F_FLASH or f >= F_OUTRO) else 2.6), kinds=(SPARK, DUST), kind_p=(0.4, 0.6), size=(2 * S, 7 * S),
           life=(1.2, 3.2) if f < F_FLASH or f >= F_OUTRO else (0.8, 1.8), light_p=0.35, layer_p=(0.05, 0.6, 0.35), squash=0.4)
    E.step(dt, drag=0.4, gravity=(0, -70 * S), turbulence=60 * S)

    if f == F_DROP:
        rose.explode(c)
        P.emit(1100, c, speed=(250 * S, 1700 * S), radius=R * 0.35, size=(5 * S, 26 * S), life=(1.2, 3.5),
               layer_p=(0.1, 0.62, 0.28), squash=0.9)
        P.emit(90, c, speed=(300 * S, 1100 * S), kinds=(SHARD, PETAL), kind_p=(0.5, 0.5), size=(14 * S, 34 * S),
               life=(1.0, 2.2), layer_p=(1.0, 0.0, 0.0), light_p=0.15)
    if F_DROP < f < F_ZOOM:
        for kf, s in KICKS:
            if kf == f:
                P.emit(int(80 * s), c, speed=(150 * S, 700 * S), radius=ring_radius(f) * 0.95, size=(4 * S, 20 * S),
                       life=(0.8, 2.0), light_p=0.3)
        if f in CLAPS:
            P.emit(50, c, speed=(400 * S, 1200 * S), kinds=(SPARK,), kind_p=(1,), size=(4 * S, 10 * S), life=(0.4, 0.9),
                   light_p=0.6)
    if F_ZOOM <= f < F_FLASH:  # flying through: streaks rushing out of the centre
        k = ramp(f, F_ZOOM, F_FLASH)
        P.emit(int(10 + 40 * k), c, speed=(900 * S, 2600 * S), kinds=(SPARK, SHARD), kind_p=(0.7, 0.3),
               size=(5 * S, 18 * S), life=(0.4, 0.9), layer_p=(0.3, 0.6, 0.1), light_p=0.35, radius=40 * S)
    if f == F_REBIRTH_END:
        P.emit(500, c, speed=(200 * S, 1100 * S), radius=R * 0.9, kinds=(SHARD, PETAL, SPARK, DUST),
               size=(4 * S, 18 * S), life=(0.6, 1.8), light_p=0.4, squash=SQ)

    orbit = c if F_DROP < f < F_ZOOM else None
    P.step(dt, drag=1.9 if f < F_ZOOM else 0.6, orbit=orbit, orbit_strength=160 * S, pull=40 * S)

    # petals
    if F_DROP <= f < F_ZOOM:
        rose.step_vortex(dt, c, strength=0.6 + 0.4 * ramp(f, F_DROP, F_DROP + 30))
    elif F_ZOOM <= f < F_FLASH:
        rose.vel += rose.off / (np.linalg.norm(rose.off, axis=1, keepdims=True) + 1) * 900 * S * dt
        rose.off += rose.vel * dt
        rose.rot += rose.rotv * dt
        rose.scl *= 1.0 + 0.9 * dt
    elif f == F_FLASH:
        a_ = rebirth_rng.uniform(0, 6.28, rose.n)
        r_ = rebirth_rng.uniform(900, 1400, rose.n) * S
        spiral_start = dict(off=np.stack([np.cos(a_) * r_, np.sin(a_) * r_], 1).astype(np.float32),
                            ang=rebirth_rng.uniform(-3, 3, rose.n))
    if F_FLASH <= f:
        k = ease_io(ramp(f, F_FLASH, F_REBIRTH_END))
        a0 = spiral_start['off']
        ang = (1 - k) * 2.2
        ca, sa = math.cos(ang), math.sin(ang)
        rot = a0 @ np.array([[ca, sa], [-sa, ca]], np.float32)
        rose.off = (rot * (1 - k)).astype(np.float32)
        rose.off[:, 1] *= 0.8
        rose.rot = (spiral_start['ang'] * (1 - k)).astype(np.float32)
        rose.scl = np.full(rose.n, 1 + 0.8 * (1 - k), np.float32)
        rose.vel[:] = 0


def ring_radius(f):
    if f < F_DROP:
        return 0.0
    grow = ease_out(ramp(f, F_DROP, F_DROP + 9))
    r = (40 + 250 * grow) * S
    return r * (1 + 0.07 * env(f, KICKS, 3.5))


# ------------------------------------------------------------------ rendering

def background(f, t):
    h2, w2 = H // 2, W // 2
    haze = nz.fbm(h2, w2, 260 * S, t * 0.35, octaves=3, offset=(t * 12, -t * 18))
    haze = cv2.resize(haze, (W, H))
    if f < F_DROP or f >= F_OUTRO:
        dusk = 0.045 + 0.04 * ramp(f, F_BUILD, F_BREAK)
        img = gradient_map(haze * dusk * 2.4, Palette.DUSK_RAMP)
        img *= (0.25 + 0.75 * soft_disc(H, W, CX, CY, W * 0.55, 1.4))[..., None]
        img += soft_disc(H, W, CX, CY + 0.25 * R, R * 2.3, 2.2)[..., None] * Palette.WINE * 0.35
    elif f < F_FLASH:  # twilight: the cold bridge behind the ring
        k = ramp(f, F_DROP, F_DROP + 12)
        img = gradient_map(np.clip(haze * 0.62 * k + 0.02, 0, 1), Palette.DUSK_RAMP)
        img *= (0.35 + 0.65 * soft_disc(H, W, CX, CY, W * 0.62, 1.2))[..., None]
    else:              # rebirth: red wash world
        img = gradient_map(np.clip(0.28 + haze * 0.25, 0, 1), Palette.RED_RAMP)
    return img.astype(np.float32)


def fire_layer(f, t, fuel_small, height, strength, rise=300):
    """toon fire at half resolution → colour → full res."""
    h2, w2 = fuel_small.shape
    fl, _ = toon_fire(h2, w2, t, fire_nz, gblur(fuel_small, 2), rise=rise * S / 2, scale=34 * S, licks=0.55,
                      height_px=height * S / 2, levels=(0.26, 0.45, 0.64))
    col = gradient_map(fl * 0.95, Palette.RED_RAMP) * (fl > 0.01)[..., None]
    col = cv2.resize(col, (W, H), interpolation=cv2.INTER_LINEAR)
    return col * strength


def render(f):
    t = f / FPS
    img = background(f, t)
    c = (CX, CY)
    kick = env(f, KICKS, 4.0)
    h2, w2 = H // 2, W // 2

    # ---- the rose
    if f < F_DROP or f >= F_FLASH:
        if f < F_BUILD:
            lit, rim = 0.2 + 0.08 * ramp(f, 0, F_BUILD), 1.05 + 0.35 * kick
        elif f < F_BREAK:
            k = ramp(f, F_BUILD, F_BREAK)
            lit, rim = 0.15 + 0.6 * k, 0.8 + 0.5 * k + 0.3 * kick
        elif f < F_DROP:
            lit, rim = 0.5 * (1 - ramp(f, F_BREAK, F_DROP - 1)), 0.7
        elif f < F_OUTRO:
            lit, rim = 0.6 + 0.4 * ramp(f, F_FLASH, F_REBIRTH_END), 1.0 + 0.4 * kick
        else:
            k = ramp(f, F_OUTRO, N - 1)
            lit, rim = 1.0 - 0.8 * k, 1.4 - 0.35 * k
        fuel = np.zeros((h2, w2), np.float32)
        open_ = 1.0 - 0.06 * ramp(f, F_BREAK, F_DROP - 1)
        bud = 1.0 if f < F_DROP else ease_out(ramp(f, F_FLASH + 18, F_REBIRTH_END))
        rose.draw(img, c, lit, rim, open_=open_, fuel=fuel, bud=bud)
        # flames from the rose (build, rebirth, outro)
        if F_BUILD <= f < F_BREAK:
            img += fire_layer(f, t, fuel * ramp(f, F_BUILD, F_BREAK - 8), 260, 0.95)
        elif F_FLASH <= f:
            s = 1.0 if f < F_OUTRO else 1 - ramp(f, F_OUTRO, N - 6)
            img += fire_layer(f, t, fuel, 320, s * (0.75 + 0.25 * kick))
        # light core in the heart (matches the ring later)
        if f < F_BREAK:
            core = 0.25 + 1.2 * ramp(f, F_BUILD, F_BREAK) ** 1.6
            img += soft_disc(H, W, CX, CY, R * (0.32 + 0.1 * kick), 2.4)[..., None] * Palette.PINK * core
            img += soft_disc(H, W, CX, CY, R * 0.12, 1.5)[..., None] * Palette.LILAC * core * 0.8
        elif f < F_DROP:  # inhale: everything darkens, a hot point charges
            k = ramp(f, F_BREAK, F_DROP - 1)
            img *= (1 - 0.75 * k)
            img += soft_disc(H, W, CX, CY, R * (0.18 - 0.12 * k), 1.2)[..., None] * Palette.LILAC * (1.2 + 2.5 * k)
        elif f >= F_OUTRO:
            k = ramp(f, F_OUTRO, N - 1)
            img += soft_disc(H, W, CX, CY, R * 0.32, 2.4)[..., None] * Palette.PINK * (0.9 * (1 - k) + 0.25 * k)
        else:
            img += soft_disc(H, W, CX, CY, R * 0.35, 2.0)[..., None] * Palette.PINK * (0.9 + 0.6 * kick)
    else:
        # exploded petals flying / orbiting (same shapes, now particles)
        rose.draw(img, c, 0.85, 1.2 + 0.4 * kick, bud=0.0)

    # ---- fire wall in the rebirth world
    if F_FLASH <= f:
        yy = np.linspace(0, 1, h2, dtype=np.float32)[:, None] * np.ones((1, w2), np.float32)
        level = 0.78 if f < F_OUTRO else 0.78 + 0.25 * ramp(f, F_OUTRO, N - 6)
        wall = smoothstep(level, level + 0.12, yy)
        s = 1.0 if f < F_OUTRO else 1 - ramp(f, F_OUTRO, N - 4)
        img += fire_layer(f, t * 0.9, wall, 560, 0.85 * s * (0.8 + 0.3 * kick), rise=380)

    # ---- the ring
    rr = ring_radius(f)
    if F_DROP <= f < F_FLASH:
        ringL = light_ring(H, W, CX, CY, rr, t * 3, nz, threads=44, jitter=0.03, thickness=max(1, int(2 * S)),
                           seed=f // 2)
        bright = 1.0 + 0.8 * kick + 2.5 * (1 - ramp(f, F_DROP, F_DROP + 8))
        img += ringL * bright
        img += soft_disc(H, W, CX, CY, rr * 1.35, 1.6)[..., None] * Palette.CRIMSON * 0.35 * bright
        if f < F_DROP + 14:  # shockwave
            k = ramp(f, F_DROP, F_DROP + 14)
            sw = np.zeros((H, W), np.float32)
            cv2.ellipse(sw, (int(CX), int(CY)), (int((80 + 1500 * ease_out(k)) * S), int((80 + 1200 * ease_out(k)) * S)),
                        0, 0, 360, 1.0, max(2, int(40 * S * (1 - k))), cv2.LINE_AA)
            img += gblur(sw, 10 * S)[..., None] * Palette.PINK * 1.4 * (1 - k)

    # ---- particles
    img = composite_particles(img, P.render(H, W, streak=1.4), fg_blur=10 * S, far_blur=1.8 * S)
    img = composite_particles(img, E.render(H, W), fg_blur=6 * S, far_blur=1.2 * S, gain=0.9)

    img = glow(img, radii=(3 * S, 12 * S, 40 * S, 120 * S), gains=(0.5, 0.42, 0.32, 0.22))

    # ---- camera
    zoom, dx, dy, rot, zb = 1.0, 0.0, 0.0, 0.0, 0.0
    if f < F_BREAK:
        zoom = 1.0 + 0.10 * ease_io(ramp(f, 0, F_BREAK))
    elif f < F_DROP:
        zoom = 1.10 - 0.04 * ramp(f, F_BREAK, F_DROP)
    elif f < F_ZOOM:
        zoom = 1.18 - 0.18 * ease_out(ramp(f, F_DROP, F_DROP + 10)) + 0.05 * ramp(f, F_DROP + 10, F_ZOOM)
        rot = 3.5 * ramp(f, F_DROP, F_ZOOM)
    elif f < F_FLASH:
        k = ramp(f, F_ZOOM, F_FLASH)
        zoom = 1.05 * math.exp(math.log(4.2 / 1.05) * k ** 2.2)
        rot = 3.5 + 10 * k ** 2
        zb = 0.02 + 0.28 * k ** 2
    elif f < F_OUTRO:
        zoom = 1.12 - 0.08 * ease_out(ramp(f, F_FLASH, F_REBIRTH_END))
    else:
        zoom = 1.04 - 0.04 * ease_io(ramp(f, F_OUTRO, N - 1))
    zoom *= 1 + 0.035 * kick
    sh = env(f, [(k, s) for k, s in KICKS if s >= 0.7 and F_DROP <= k < F_OUTRO], 1.2)
    if sh > 0.05:
        r_ = np.random.default_rng(f)
        dx, dy = r_.normal(0, 7 * S * sh), r_.normal(0, 5 * S * sh)
    img = camera(img, zoom, dx, dy, rot, center=(CX, CY))
    if zb > 0:
        img = zoom_blur(img, zb, (CX, CY), samples=12)

    # ---- transitions
    if f == F_DROP - 1:  # the inhale: one black frame
        img[:] = 0
    wo = {F_DROP: 1.0, F_DROP + 1: 0.85, F_DROP + 2: 0.55, F_DROP + 3: 0.3, F_DROP + 4: 0.12}.get(f, 0.0)
    wo = max(wo, {F_FLASH - 2: 0.35, F_FLASH - 1: 0.75, F_FLASH: 1.0, F_FLASH + 1: 0.7, F_FLASH + 2: 0.4,
                  F_FLASH + 3: 0.15}.get(f, 0.0))
    if wo:
        tint = Palette.LILAC if f < F_FLASH - 2 or f > F_FLASH + 3 else (Palette.LILAC * 0.6 + Palette.PINK * 0.4)
        img = img * (1 - wo) + (tint * 1.15) * wo + img * wo * 0.3
    if f in (F_REBIRTH_END, F_REBIRTH_END + 1):  # impact frame: inverted, high contrast
        l = np.clip(luma(img), 0, 1)
        img = gradient_map(1 - l ** 0.8, [(0.0, '#0D050F'), (0.35, '#4D091E'), (0.7, '#E31E3C'), (1.0, '#E6D9E7')])
        img = img * 1.05

    ca = 1.6 * S + 5 * S * kick + (8 * S if f in (F_DROP, F_DROP + 1, F_FLASH) else 0)
    return finish(img, ca=ca, grain_amt=0.026, vig=0.5, seed=f, halation=0.3)


# ------------------------------------------------------------------ main

only = [int(x) for x in A.only.split(',') if x]
os.makedirs(os.path.join(A.out, 'frames'), exist_ok=True)
writer = None
if not only:
    name = 'burning_rose.mp4' if DIV == 1 else f'preview_{DIV}.mp4'
    writer = FFmpegWriter(os.path.join(A.out, name), W, H, FPS, crf=14, audio=os.path.join(A.out, 'track.wav'))
for _ in range(90):  # warm-up: embers already drifting at frame 0 (matches the loop's end)
    E.emit(3, (CX, CY + 40 * S), speed=(30 * S, 150 * S), spread=(-np.pi * 0.85, -np.pi * 0.15), radius=R,
           kinds=(SPARK, DUST), kind_p=(0.4, 0.6), size=(2 * S, 7 * S), life=(1.2, 3.2), light_p=0.35,
           layer_p=(0.05, 0.6, 0.35), squash=0.4)
    E.step(1 / FPS, drag=0.4, gravity=(0, -70 * S), turbulence=60 * S)
cache = {}
t0 = time.time()
for f in range(N):
    simulate(f)
    key = HOLDS.key(f)
    if only:
        if f in only:
            out = render(f)
            cv2.imwrite(os.path.join(A.out, 'frames', f'f_{f:04d}.png'), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            print('frame', f, f'{time.time() - t0:.1f}s', flush=True)
        if f >= max(only):
            break
        continue
    if key == f:
        cache = {f: render(f)}
    writer.write(cache[key])
    if f % 25 == 0:
        print(f'frame {f}/{N}  {time.time() - t0:.0f}s', flush=True)
if writer:
    writer.close()
    print(f'done in {time.time() - t0:.0f}s; effective fps {HOLDS.effective_fps():.1f}')
