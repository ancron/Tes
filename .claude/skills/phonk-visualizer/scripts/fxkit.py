"""fxkit — code-driven 2D FX compositing for dark phonk visualizers.

Float32 RGB frames in [0, 1+] (values above 1 are "hot" light, soft-clipped at the end).
Everything is plain numpy + OpenCV so a frame can be rendered, saved and inspected alone.

Main pieces
  Palette / gradient_map      — palette discipline (build in luma, map to colours)
  Noise.fbm / displace        — animated fractal noise, Turbulent-Displace analogue
  toon_fire                   — stylised 2D fire field (posterised bands, licks, C-cuts)
  glow / zoom_blur            — multi-scale bloom, radial zoom blur
  light_ring                  — Saber-like ring of jittering threads
  Particles                   — 3-tier particle system (shards, petals, sparks, dust)
  camera / chroma_aberration / grain / vignette / finish — camera moves and texture
  HoldSchedule                — "Posterize Time": which frames are unique, holds on 1s/2s/3s
  BeatGrid                    — BPM → frame numbers (cumulative rounding)
  FFmpegWriter                — stream frames to an H.264 file
"""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------- palette

def hex2rgb(h: str) -> np.ndarray:
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


class Palette:
    """Canonical crimson palette measured from the genre reference (see style-bible.md)."""
    VOID = hex2rgb('#0D050F')      # violet-black base
    WINE = hex2rgb('#4D091E')
    BLOOD = hex2rgb('#95122E')
    CRIMSON = hex2rgb('#E31E3C')
    PINK = hex2rgb('#E66889')
    LILAC = hex2rgb('#E6D9E7')     # light — never pure white
    DUSK = hex2rgb('#392D46')      # cold bridge
    MAUVE = hex2rgb('#785D79')
    RED_RAMP = [(0.0, '#0D050F'), (0.22, '#4D091E'), (0.42, '#95122E'), (0.62, '#E31E3C'),
                (0.82, '#E66889'), (1.0, '#E6D9E7')]
    DUSK_RAMP = [(0.0, '#0D050F'), (0.35, '#392D46'), (0.7, '#785D79'), (1.0, '#E6D9E7')]


def gradient_map(luma: np.ndarray, stops, n: int = 1024) -> np.ndarray:
    """Map a [0,1] luminance image to colours along `stops` [(pos, '#hex'), ...]."""
    pos = np.array([p for p, _ in stops], np.float32)
    cols = np.array([hex2rgb(c) for _, c in stops], np.float32)
    xs = np.linspace(0, 1, n, dtype=np.float32)
    lut = np.stack([np.interp(xs, pos, cols[:, k]) for k in range(3)], 1).astype(np.float32)
    idx = np.clip(luma * (n - 1), 0, n - 1).astype(np.int32)
    return lut[idx]


def luma(img: np.ndarray) -> np.ndarray:
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- noise

class Noise:
    """Animated value-noise fBm. Each octave is a random lattice interpolated in time
    (smoothstep between integer time slices) and upsampled with bicubic resize."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._cache: dict = {}

    P = 160  # periodic lattice size (cells) — scrolling wraps seamlessly on the torus

    def _lattice(self, octave, k):
        key = (octave, k)
        g = self._cache.get(key)
        if g is None:
            rng = np.random.default_rng((self.seed * 1_000_003 + octave * 7919 + k * 104_729) & 0x7FFFFFFF)
            g = rng.random((self.P, self.P), dtype=np.float32)
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[key] = g
        return g

    def fbm(self, h, w, scale, t=0.0, octaves=4, lacunarity=2.0, gain=0.5, speed=1.0, offset=(0.0, 0.0)):
        """Noise in [0,1], shape (h, w). `scale` = feature size in pixels of the first octave.
        `t` evolves the pattern; `offset` (pixels) scrolls it (e.g. offset=(0, t*speed) to rise)."""
        out = np.zeros((h, w), np.float32)
        amp, norm, cell = 1.0, 0.0, float(scale)
        for o in range(octaves):
            c = max(2, int(round(cell)))
            gh, gw = int(math.ceil(h / c)) + 3, int(math.ceil(w / c)) + 3
            to = t * speed * (1 + 0.37 * o)
            k = math.floor(to)
            s = to - k
            s = s * s * (3 - 2 * s)
            ofx, ofy = (offset[0] + o * 977.0) / c, (offset[1] + o * 613.0) / c
            ix0, iy0 = math.floor(ofx), math.floor(ofy)
            rows = (iy0 + np.arange(gh)) % self.P
            cols = (ix0 + np.arange(gw)) % self.P
            a = self._lattice(o, k)[np.ix_(rows, cols)]
            b = self._lattice(o, k + 1)[np.ix_(rows, cols)]
            g = ((a * (1 - s) + b * s) - 0.5) / math.sqrt((1 - s) ** 2 + s ** 2) + 0.5  # keep contrast while blending
            up = cv2.resize(g, (gw * c, gh * c), interpolation=cv2.INTER_CUBIC)
            px, py = int((ofx - ix0) * c), int((ofy - iy0) * c)
            out += amp * up[py:py + h, px:px + w]
            norm += amp
            amp *= gain
            cell /= lacunarity
        return np.clip((out / norm - 0.5) * 1.6 + 0.5, 0, 1)


def displace(img: np.ndarray, dx: np.ndarray, dy: np.ndarray, amp: float) -> np.ndarray:
    """Turbulent-Displace analogue: dx, dy are [0,1] noise fields, amp in pixels."""
    h, w = dx.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx = xx + (dx - 0.5) * 2 * amp
    my = yy + (dy - 0.5) * 2 * amp
    return cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------- blur / light

def gblur(img, sigma):
    if sigma <= 0.3:
        return img
    if sigma > 12:  # blur a downscaled copy for speed on big radii
        f = int(sigma // 6)
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma / f)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(img, (0, 0), sigma)


def glow(img, radii=(3, 12, 40, 110), gains=(0.55, 0.45, 0.35, 0.25), threshold=0.55, tint=None):
    """Deep-Glow-like multi-scale bloom of the bright part of `img`, added on top."""
    l = luma(img)
    bright = img * smoothstep(threshold, threshold + 0.35, l)[..., None]
    acc = np.zeros_like(img)
    for r, g in zip(radii, gains):
        acc += gblur(bright, r) * g
    if tint is not None:
        acc = acc * 0.5 + luma(acc)[..., None] * np.asarray(tint, np.float32) * 0.5
    return img + acc


def zoom_blur(img, strength=0.08, center=None, samples=10):
    """Radial (zoom) blur: average of copies scaled about `center` up to 1+strength."""
    if strength <= 1e-4:
        return img
    h, w = img.shape[:2]
    cx, cy = center if center is not None else (w / 2, h / 2)
    acc = np.zeros_like(img)
    for i in range(samples):
        s = 1 + strength * i / (samples - 1)
        M = np.float32([[s, 0, cx - s * cx], [0, s, cy - s * cy]])
        acc += cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return acc / samples


def light_ring(h, w, cx, cy, r, t, noise: Noise, threads=36, jitter=0.035, thickness=1, squash=1.0,
               color=None, seed=0):
    """Saber-like ring: many thin jittering ellipses. Returns an additive light layer (h,w,3)."""
    color = Palette.LILAC if color is None else np.asarray(color, np.float32)
    layer = np.zeros((h, w), np.float32)
    rng = np.random.default_rng(seed)
    n = 220
    ang = np.linspace(0, 2 * np.pi, n, endpoint=True)
    for i in range(threads):
        ph = rng.random() * 6.283
        fr = 2 + rng.integers(0, 5)
        wob = (np.sin(ang * fr + ph + t * (1.3 + rng.random() * 2)) * 0.6
               + np.sin(ang * (fr + 3) - ph * 1.7 + t * 2.1) * 0.4)
        rr = r * (1 + jitter * wob * (0.4 + rng.random()))
        pts = np.stack([cx + rr * np.cos(ang), cy + rr * np.sin(ang) * squash], 1)
        val = 0.25 + 0.75 * rng.random()
        tmp = np.zeros((h, w), np.uint8)
        cv2.polylines(tmp, [np.round(pts * 4).astype(np.int32)], False, int(255 * val), thickness,
                      cv2.LINE_AA, shift=2)
        layer += tmp.astype(np.float32) / 255.0 * 0.22
    layer = np.clip(layer, 0, 2.5)
    return layer[..., None] * color[None, None, :]


def soft_disc(h, w, cx, cy, r, power=2.0):
    """Radial falloff 1 at centre → 0 at r (for light cores, vignettes, masks)."""
    yy, xx = np.ogrid[:h, :w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(r, 1e-6)
    return np.clip(1 - d, 0, 1) ** power


# ---------------------------------------------------------------- stylised fire

def toon_fire(h, w, t, noise: Noise, base_mask: np.ndarray, rise=260.0, scale=70.0, levels=(0.30, 0.48, 0.66),
              licks=0.6, height_px=420, stretch=3.2, seed_off=(0, 0)):
    """Stylised 2D fire as posterised bands.

    base_mask: (h,w) [0,1] — where the fuel is. Heat is smeared upward (height_px), then eaten
    by vertically stretched rising noise → tapering tongues; small noise bites C-cuts/hooks.
    rise = scroll speed px/s, scale = tongue width px, licks = sideways sway amount.
    Returns (fire_luma in {0..1} steps, soft_luma) — map with gradient_map for colour.
    """
    # heat: fuel smeared upward with decay → 1 at the fuel, fading with height above it
    heat = base_mask.astype(np.float32).copy()
    nsteps = 24
    step = max(1, int(height_px / nsteps))
    shifted = heat.copy()
    for i in range(1, nsteps + 1):
        shifted = np.vstack([shifted[step:], np.zeros((step, w), np.float32)])
        heat = np.maximum(heat, shifted * (1 - i / (nsteps + 1)) ** 1.15)
    # vertically stretched noise scrolling upward → tongue shapes (thick base, pointed tips)
    hs = max(8, int(h / stretch))
    n1 = noise.fbm(hs, w, scale, t * 1.3, octaves=3, gain=0.45,
                   offset=(seed_off[0], seed_off[1] + t * rise / stretch))
    n1 = cv2.resize(n1, (w, h), interpolation=cv2.INTER_CUBIC)
    n2 = noise.fbm(h, w, scale * 0.6, t * 2.2, octaves=2, offset=(seed_off[0] + 333, seed_off[1] + t * rise * 1.4))
    # S-sway: horizontal displacement (licks bend like in wind)
    sway = noise.fbm(h, w, scale * 4, t * 0.6, octaves=2, offset=(seed_off[0] + 71, seed_off[1] + t * rise * 0.3))
    heat_w = displace(heat, sway, np.full_like(sway, 0.5), amp=scale * licks)
    # tongues survive higher where noise is strong (power → pointed tips); a little small noise
    # bites C-cuts and hooks into the edges; blur keeps the nested bands clean, not speckled
    field = heat_w ** 1.35 * (0.2 + 1.15 * n1) - 0.12 * (1 - n2) * heat_w
    field = cv2.GaussianBlur(field, (0, 0), max(1.5, scale * 0.07))
    soft = np.clip(field, 0, 1)
    out = np.zeros_like(soft)
    for i, lv in enumerate(levels):
        out = np.where(soft > lv, (i + 1) / len(levels), out)
    # slightly soften band edges so they glow rather than alias
    out = cv2.GaussianBlur(out, (0, 0), 1.2)
    return out, soft


# ---------------------------------------------------------------- particles

SHARD, PETAL, SPARK, DUST = 0, 1, 2, 3


@dataclass
class Particles:
    """3-tier particle system. Layers: 0 = foreground (big, defocused), 1 = mid (crisp), 2 = far (small, dim)."""
    seed: int = 0
    pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    vel: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    size: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    rot: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    rotv: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    age: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    life: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    kind: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    layer: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    light: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))  # 0 = crimson, 1 = lilac light
    shape_seed: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def emit(self, n, center, speed=(200, 900), spread=(0, 2 * np.pi), radius=0.0, kinds=(SHARD, PETAL, SPARK, DUST),
             kind_p=(0.35, 0.25, 0.25, 0.15), size=(4, 26), life=(0.8, 2.5), light_p=0.25, layer_p=(0.12, 0.6, 0.28),
             squash=1.0):
        r = self.rng
        a = r.uniform(*spread, n)
        sp = r.uniform(*speed, n) * (0.4 + 0.6 * r.random(n) ** 0.5)
        rad = radius * np.sqrt(r.random(n))
        p = np.stack([center[0] + np.cos(a) * rad, center[1] + np.sin(a) * rad * squash], 1)
        v = np.stack([np.cos(a) * sp, np.sin(a) * sp * squash], 1)
        k = r.choice(kinds, n, p=np.array(kind_p) / np.sum(kind_p))
        lay = r.choice(3, n, p=layer_p)
        s = r.uniform(*size, n) * np.where(lay == 0, 3.2, np.where(lay == 2, 0.45, 1.0))
        s = np.where(k == DUST, s * 0.25, np.where(k == SPARK, s * 0.5, s))
        self._append(p, v, s, r.uniform(0, 6.28, n), r.uniform(-8, 8, n), np.zeros(n), r.uniform(*life, n), k, lay,
                     (r.random(n) < light_p).astype(np.float32), r.integers(0, 1 << 30, n))

    def _append(self, *arrs):
        names = ['pos', 'vel', 'size', 'rot', 'rotv', 'age', 'life', 'kind', 'layer', 'light', 'shape_seed']
        for nm, a in zip(names, arrs):
            cur = getattr(self, nm)
            setattr(self, nm, np.concatenate([cur, np.asarray(a, cur.dtype)]) if len(cur) else np.asarray(a, cur.dtype))

    def step(self, dt, drag=1.6, gravity=(0, 0), orbit=None, orbit_strength=0.0, pull=0.0, turbulence=0.0):
        """orbit=(cx,cy): adds tangential swirl (orbit_strength px/s) and radial pull toward the centre."""
        if not len(self.pos):
            return
        self.vel *= math.exp(-drag * dt)
        self.vel += np.asarray(gravity, np.float32) * dt
        if orbit is not None:
            d = self.pos - np.asarray(orbit, np.float32)
            dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-3
            tang = np.concatenate([-d[:, 1:2], d[:, 0:1]], 1) / dist
            self.vel += tang * orbit_strength * dt * 3 - d / dist * pull * dt
        if turbulence:
            self.vel += self.rng.normal(0, turbulence, self.vel.shape).astype(np.float32) * dt
        self.pos += self.vel * dt
        self.rot += self.rotv * dt
        self.age += dt
        keep = self.age < self.life
        for nm in ['pos', 'vel', 'size', 'rot', 'rotv', 'age', 'life', 'kind', 'layer', 'light', 'shape_seed']:
            setattr(self, nm, getattr(self, nm)[keep])

    def render(self, h, w, dt=1 / 30, streak=1.0):
        """Returns three additive RGB layers (fg, mid, far) — blur fg/far yourself (depth of field)."""
        layers = [np.zeros((h, w, 3), np.float32) for _ in range(3)]
        for i in range(len(self.pos)):
            fade = min(1.0, (self.life[i] - self.age[i]) / 0.35) * min(1.0, self.age[i] / 0.05 + 0.3)
            col = Palette.LILAC if self.light[i] > 0.5 else (Palette.CRIMSON, Palette.CRIMSON, Palette.BLOOD, Palette.PINK)[self.shape_seed[i] % 4]
            lay = int(self.layer[i])
            if lay == 2:
                fade *= 0.55
            pts = self._shape(i, dt, streak)
            if pts is None:
                continue
            c = tuple(float(x) * fade for x in col)
            cv2.fillPoly(layers[lay], [np.round(pts * 4).astype(np.int32)], c, cv2.LINE_AA, shift=2)
        return layers

    def _shape(self, i, dt, streak):
        x, y = self.pos[i]
        s = float(self.size[i])
        k = int(self.kind[i])
        vx, vy = self.vel[i]
        sp = math.hypot(vx, vy)
        ang = math.atan2(vy, vx)
        stretch = 1.0 + min(6.0, sp * dt * streak / max(s, 1.0))
        rr = np.random.default_rng(int(self.shape_seed[i]))
        if k == SHARD:  # jagged 3–6 vertex polygon
            m = int(rr.integers(3, 7))
            a = np.sort(rr.uniform(0, 2 * np.pi, m)) + self.rot[i]
            rad = s * rr.uniform(0.35, 1.0, m)
            local = np.stack([np.cos(a) * rad, np.sin(a) * rad], 1)
        elif k == PETAL:  # lobe with a notch, rotating
            a = np.linspace(0, 2 * np.pi, 18, endpoint=False)
            rad = s * (0.55 + 0.45 * np.cos(a)) * (1 - 0.18 * np.exp(-((a - np.pi) / 0.4) ** 2))
            local = np.stack([np.cos(a) * rad * 1.1, np.sin(a) * rad * 0.55], 1)
            cr, sr = math.cos(self.rot[i]), math.sin(self.rot[i])
            local = local @ np.array([[cr, sr], [-sr, cr]], np.float32)
        elif k == SPARK:  # streak along velocity
            L = s * stretch * 1.8
            local = np.array([[-L, -s * 0.18], [L * 0.2, -s * 0.3], [L * 0.2, s * 0.3], [-L, s * 0.18]], np.float32)
            ca, sa = math.cos(ang), math.sin(ang)
            return local @ np.array([[ca, sa], [-sa, ca]], np.float32) + np.array([x, y], np.float32)
        else:  # dust: tiny disc
            a = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            local = np.stack([np.cos(a), np.sin(a)], 1) * max(0.8, s)
        if stretch > 1.15 and k != DUST:  # motion stretch along velocity
            ca, sa = math.cos(ang), math.sin(ang)
            R = np.array([[ca, sa], [-sa, ca]], np.float32)
            loc = local @ R.T
            loc[:, 0] *= stretch
            local = loc @ R
        return local.astype(np.float32) + np.array([x, y], np.float32)


def composite_particles(img, layers, fg_blur=9.0, far_blur=1.5, gain=1.0):
    fg, mid, far = layers
    return img + (gblur(far, far_blur) + mid + gblur(fg, fg_blur) * 0.8) * gain


# ---------------------------------------------------------------- camera & texture

def camera(img, zoom=1.0, dx=0.0, dy=0.0, rot_deg=0.0, center=None):
    h, w = img.shape[:2]
    cx, cy = center if center is not None else (w / 2, h / 2)
    M = cv2.getRotationMatrix2D((cx, cy), rot_deg, zoom)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def chroma_aberration(img, px=2.0):
    if px <= 0.05:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    for ch, s in ((0, 1 + px / w * 2), (2, 1 - px / w * 2)):
        M = np.float32([[s, 0, w / 2 * (1 - s)], [0, s, h / 2 * (1 - s)]])
        out[..., ch] = cv2.warpAffine(img[..., ch], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def vignette(img, strength=0.35, power=2.2):
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / math.sqrt(2)
    return img * (1 - strength * d ** power)[..., None]


def grain(img, amount=0.03, seed=0):
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    n = rng.normal(0, 1, (h // 2 + 1, w // 2 + 1)).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    return img + n * amount * (0.35 + 0.65 * np.sqrt(np.clip(luma(img), 0, 1)))[..., None]


def soft_clip(img, knee=0.8):
    """Filmic shoulder: linear below `knee`, smooth roll-off to 1 above (keeps hot light from banding)."""
    over = np.maximum(img - knee, 0)
    return np.where(img > knee, knee + (1 - knee) * (1 - np.exp(-over / (1 - knee))), img)


def finish(img, base=Palette.VOID, ca=1.5, grain_amt=0.03, vig=0.35, seed=0, halation=0.25):
    """Final look: halation, violet black point, vignette, CA, grain, soft clip → uint8 RGB."""
    if halation:
        hot = np.maximum(img - 0.75, 0)
        img = img + gblur(hot, 18) * np.array([1.0, 0.25, 0.35], np.float32) * halation
    img = vignette(img, vig)
    img = chroma_aberration(img, ca)
    img = base + img * (1 - base)  # violet-black floor, never neutral black
    img = grain(img, grain_amt, seed)
    img = soft_clip(img)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------- timing

class HoldSchedule:
    """Posterize-Time. segments: [(start_frame, end_frame_exclusive, hold), ...] covering the timeline.
    key(f) → the frame whose image is shown at f (first frame of its hold group)."""

    def __init__(self, segments, total):
        self.key_of = list(range(total))
        for s, e, hold in segments:
            for f in range(s, min(e, total)):
                self.key_of[f] = s + ((f - s) // hold) * hold

    def key(self, f):
        return self.key_of[f]

    def unique(self):
        return sorted(set(self.key_of))

    def effective_fps(self, fps=30):
        return len(self.unique()) / (len(self.key_of) / fps)


class BeatGrid:
    def __init__(self, bpm, fps=30, offset_s=0.0, beats_per_bar=4):
        self.bpm, self.fps, self.offset, self.bpb = bpm, fps, offset_s, beats_per_bar
        self.spb = 60.0 / bpm

    def beat(self, i):  # cumulative rounding — no drift
        return int(round((self.offset + i * self.spb) * self.fps))

    def bar(self, i):
        return self.beat(i * self.bpb)

    def time(self, f):
        return f / self.fps


def pulse(f, hits, decay=5.0, attack=0):
    """Envelope 1 at each hit frame decaying exponentially over `decay` frames (0 before the hit)."""
    v = 0.0
    for hf in hits:
        d = f - hf
        if -attack <= d < 6 * decay:
            v = max(v, math.exp(-max(d, 0) / decay) if d >= 0 else (d + attack + 1) / (attack + 1))
    return v


# ---------------------------------------------------------------- output

class FFmpegWriter:
    def __init__(self, path, w, h, fps=30, crf=15, audio=None, preset='slow'):
        cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
               '-r', str(fps), '-i', '-']
        if audio:
            cmd += ['-i', audio, '-c:a', 'aac', '-b:a', '320k', '-shortest']
        cmd += ['-c:v', 'libx264', '-preset', preset, '-crf', str(crf), '-pix_fmt', 'yuv420p',
                '-profile:v', 'high', '-movflags', '+faststart', path]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, rgb_u8):
        self.p.stdin.write(np.ascontiguousarray(rgb_u8).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()
