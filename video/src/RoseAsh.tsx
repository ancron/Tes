import React from "react";
import { AbsoluteFill, Freeze, Img, random, Sequence, staticFile, useCurrentFrame } from "remotion";
import { Audio, Video } from "@remotion/media";
import { zoomBlur } from "@remotion/effects/zoom-blur";

// "Rose / Ash" v4 — see video/PLAN.md. Critic round 1 → constructed VFX events (shatter, inhale),
// 3–5 frame cold flashes with the next scene born inside, mixed 1/2/3/4 cadence, no idle jitter.
export const DURATION = 362;
const W = 1920;
const H = 1080;

const KICKS = [9, 32, 56, 79, 102, 149, 173, 196, 220, 243, 266, 290, 313, 337, 360];
const POSES = [22, 44, 67, 90, 153, 185, 231, 256, 302, 325]; // pose jumps between kicks

// Posterize-time with a living cadence: hold 3–4 right after a hit, ones on flashes/velocity.
const SEGS: [number, number, number][] = [
  [0, 3, 3], [3, 9, 2], [9, 12, 1], [12, 15, 3], [15, 22, 2], [22, 25, 3], [25, 32, 2], [32, 35, 3], [35, 44, 2],
  [44, 47, 3], [47, 56, 2], [56, 60, 4], [60, 67, 2], [67, 70, 3], [70, 79, 2], [79, 82, 3], [82, 90, 2], [90, 93, 3],
  [93, 98, 2], [98, 102, 1], [102, 105, 1], [105, 108, 3], [108, 122, 2], [122, 134, 1], [134, 146, 3], [146, 149, 1],
  [149, 153, 1], [153, 161, 1], [161, 173, 2], [173, 176, 3], [176, 185, 2], [185, 188, 3], [188, 196, 2],
  [196, 199, 1], [199, 202, 3], [202, 220, 2], [220, 223, 3], [223, 231, 2], [231, 234, 3], [234, 243, 2],
  [243, 246, 1], [246, 249, 3], [249, 256, 2], [256, 259, 3], [259, 266, 2], [266, 269, 3], [269, 290, 2], [290, 292, 1], [292, 295, 3],
  [295, 302, 2], [302, 305, 3], [305, 313, 2], [313, 317, 4], [317, 325, 2], [325, 328, 3], [328, 337, 2],
  [337, 340, 3], [340, 357, 2], [357, 362, 1],
];

export const drawingOf = (f: number) => {
  const s = SEGS.find(([a, b]) => f >= a && f < b) ?? SEGS[SEGS.length - 1];
  return s[0] + Math.floor((f - s[0]) / s[2]) * s[2];
};

const hitAt = (d: number) => {
  let v = 0;
  for (const k of KICKS) if (d >= k && d - k < 10) v = Math.max(v, Math.pow(0.5, (d - k) / 2));
  for (const k of POSES) if (d >= k && d - k < 6) v = Math.max(v, 0.55 * Math.pow(0.5, (d - k) / 2));
  return v;
};

// flashes: red flood, cold white, or a burst from a light source; BIRTH = the next scene inside the white
type Flash = { kind: "red" | "white" | "burst"; a?: number; x?: number; y?: number };
const FLASH: Record<number, Flash> = {
  9: { kind: "red", a: 1 }, 10: { kind: "red", a: 0.8 },
  102: { kind: "burst", x: 0.56, y: 0.4 }, 103: { kind: "white", a: 1 },
  149: { kind: "burst", x: 0.5, y: 0.5 }, 150: { kind: "white", a: 1 }, 151: { kind: "white", a: 1 },
  196: { kind: "burst", x: 0.5, y: 0.28 }, 197: { kind: "white", a: 1 },
  243: { kind: "red", a: 1 }, 244: { kind: "red", a: 0.8 },
  357: { kind: "burst", x: 0.5, y: 0.5 }, 358: { kind: "white", a: 1 }, 359: { kind: "white", a: 1 },
  360: { kind: "white", a: 1 },
};
const BIRTH = new Set([11, 104, 152, 198, 245, 361]);
const IMPACT = new Set([290, 291]);

// ---------------------------------------------------------------- layers

type ClipProps = {
  id: string;
  at: number;
  rate?: number;
  mirror?: boolean;
  scale?: number;
  x?: number;
  y?: number;
  blend?: React.CSSProperties["mixBlendMode"];
  opacity?: number;
  filter?: string;
  zoom?: (local: number) => number;
};

const Clip: React.FC<ClipProps> = (p) => {
  const local = useCurrentFrame();
  const zb = p.zoom ? p.zoom(local) : 0;
  const s = p.scale ?? 1;
  return (
    <AbsoluteFill
      style={{
        mixBlendMode: p.blend ?? "normal",
        opacity: p.opacity ?? 1,
        filter: p.filter,
        transform: `translate(${p.x ?? 0}px, ${p.y ?? 0}px) scale(${(p.mirror ? -1 : 1) * s}, ${s})`,
      }}
    >
      <Video
        src={staticFile(`footage/graded/${p.id}.mp4`)}
        trimBefore={Math.round(p.at * 30)}
        playbackRate={p.rate ?? 1}
        muted
        objectFit="cover"
        effects={zb > 0.5 ? [zoomBlur({ amount: zb, center: [0.5, 0.5], samples: 64 })] : []}
        style={{ width: W, height: H }}
      />
    </AbsoluteFill>
  );
};

const Shot: React.FC<{ from: number; to: number; name: string; children: React.ReactNode }> = ({
  from,
  to,
  name,
  children,
}) => (
  <Sequence from={from} durationInFrames={to - from} name={name} layout="none">
    <AbsoluteFill>{children}</AbsoluteFill>
  </Sequence>
);

const Point: React.FC<{ size: number; power: number; x?: number; y?: number }> = ({ size, power, x = 0.5, y = 0.5 }) => (
  <AbsoluteFill style={{ mixBlendMode: "screen" }}>
    <div
      style={{
        position: "absolute",
        left: x * W - size * 2,
        top: y * H - size * 2,
        width: size * 4,
        height: size * 4,
        borderRadius: "50%",
        background: `radial-gradient(circle, rgba(255,250,252,${Math.min(1, power)}) 0%, rgba(255,236,240,${
          0.9 * power
        }) 12%, rgba(227,30,60,${0.55 * power}) 30%, rgba(110,8,24,${0.25 * power}) 55%, rgba(0,0,0,0) 75%)`,
      }}
    />
  </AbsoluteFill>
);

// a big defocused rose petal sweeping past the lens (fly-through foreground)
const ForegroundPass: React.FC = () => {
  const t = useCurrentFrame(); // 0..7
  const x = -1700 + t * 480;
  return (
    <AbsoluteFill style={{ transform: `translate(${x}px, ${60 - t * 30}px) rotate(${-18 + t * 6}deg) scale(3.4)` }}>
      <Clip id="171" at={6.0} filter="blur(9px) brightness(1.3)" opacity={0.95} />
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------- camera (poses only on hits)

const Camera: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const d = useCurrentFrame();
  const hit = hitAt(d);
  const r = (k: string) => random(`${k}-${d}`) * 2 - 1;
  let scale = 1.05 + 0.1 * hit;
  let tx = 38 * hit * r("hx");
  let ty = 26 * hit * r("hy");
  let rot = 1.4 * hit * r("hr");
  if (d >= 122 && d < 134) {
    const k = (d - 122) / 11;
    scale = 1.05 * Math.pow(3.4, k * k);
    tx = 0;
    ty = 0;
    rot = 7 * k * k;
  }
  let bright = 1 + 0.18 * r("e") + 0.38 * hit;
  if (d >= 134 && d < 149) bright = 1;
  return (
    <AbsoluteFill
      style={{
        transform: `translate(${tx}px, ${ty}px) rotate(${rot}deg) scale(${scale})`,
        filter: `brightness(${bright.toFixed(3)}) contrast(${(1.06 + 0.08 * hit).toFixed(3)})`,
      }}
    >
      {children}
    </AbsoluteFill>
  );
};

const FlashLayer: React.FC<{ f: Flash }> = ({ f }) => {
  if (f.kind === "white") return <AbsoluteFill style={{ backgroundColor: `rgba(238,235,246,${f.a ?? 1})` }} />;
  if (f.kind === "red")
    return (
      <AbsoluteFill
        style={{
          background: `radial-gradient(ellipse at center, rgba(255,246,244,${f.a}) 0%, rgba(255,90,100,${
            f.a
          }) 28%, rgba(227,30,60,${f.a}) 60%, rgba(150,10,35,${f.a}) 100%)`,
        }}
      />
    );
  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(circle at ${(f.x ?? 0.5) * 100}% ${
          (f.y ?? 0.5) * 100
        }%, rgba(246,244,252,1) 0%, rgba(236,232,246,0.97) 30%, rgba(227,60,80,0.55) 55%, rgba(0,0,0,0) 85%)`,
      }}
    />
  );
};

// ---------------------------------------------------------------- the edit (rendered on drawings)

const Edit: React.FC = () => {
  const d = useCurrentFrame();
  const dive = (local: number) => (local >= 18 ? 10 + 14 * (local - 18) : 0); // F starts at 104 → dive from 122
  const birth = BIRTH.has(d);
  const impact = IMPACT.has(d);
  const sceneFilter = birth
    ? "grayscale(1) invert(1) contrast(1.35) brightness(1.04)"
    : impact
      ? "grayscale(1) brightness(2) contrast(2.5) invert(1)"
      : undefined;
  const flash = FLASH[d];

  return (
    <AbsoluteFill style={{ backgroundColor: "black" }}>
      <Camera>
        <AbsoluteFill style={{ filter: sceneFilter }}>
          <Shot name="A smoke" from={0} to={11}><Clip id="50951" at={1.867} rate={2.0} /></Shot>
          <Shot name="B rose" from={11} to={22}><Clip id="171" at={2.0} rate={3.0} scale={1.15} /></Shot>
          <Shot name="B rose 2" from={22} to={32}><Clip id="171" at={5.5} rate={3.0} scale={1.35} x={160} mirror /></Shot>
          <Shot name="C hands of fire" from={32} to={44}><Clip id="40938f" at={4.5} rate={1.8} scale={1.1} /></Shot>
          <Shot name="C hands of fire 2" from={44} to={56}><Clip id="40938f" at={7.5} rate={1.8} mirror scale={1.25} /></Shot>
          <Shot name="D watcher" from={56} to={67}><Clip id="1038" at={1.2} rate={1.6} scale={1.2} /></Shot>
          <Shot name="D watcher 2" from={67} to={79}><Clip id="1038" at={6.0} rate={1.6} scale={1.35} /></Shot>
          <Shot name="E burning figure" from={79} to={90}>
            <Clip id="1038" at={10.8} rate={1.6} scale={1.2} x={110} y={-40} />
            <Clip id="3759" at={4.0} rate={1.5} blend="screen" scale={1.2} y={200} />
          </Shot>
          <Shot name="E burning figure 2" from={90} to={104}>
            <Clip id="1038" at={13.2} rate={1.6} scale={1.25} />
            <Clip id="3759" at={6.0} rate={1.5} blend="screen" scale={1.2} y={160} />
          </Shot>
          <Shot name="F macro + dive" from={104} to={134}>
            <Clip id="100917" at={1.0} rate={1.6} zoom={dive} />
            <Clip id="3759" at={9.0} rate={1.5} blend="screen" scale={1.1} y={80} />
          </Shot>
          <Shot name="F fly-through petal" from={127} to={134}><ForegroundPass /></Shot>
          <Shot name="G inhale" from={134} to={149}><Clip id="vfx_inhale" at={0} rate={0.95} /></Shot>
          <Shot name="H shatter" from={152} to={173}><Clip id="vfx_shatter" at={0} rate={0.95} /></Shot>
          <Shot name="I dancer" from={173} to={185}><Clip id="33899" at={8.4} rate={1.4} scale={1.15} /></Shot>
          <Shot name="I dancer 2" from={185} to={198}><Clip id="33899" at={11.0} rate={1.4} scale={1.35} mirror /></Shot>
          <Shot name="J arms up" from={198} to={220}><Clip id="33899" at={14.0} rate={1.3} scale={1.4} y={80} mirror /></Shot>
          <Shot name="K cold burning hands" from={220} to={231}><Clip id="40938cf" at={7.1} rate={1.6} /></Shot>
          <Shot name="K cold burning hands 2" from={231} to={245}><Clip id="40938cf" at={9.6} rate={1.6} mirror scale={1.25} /></Shot>
          <Shot name="L roses on fire" from={245} to={256}>
            <Clip id="100898" at={3.0} rate={1.8} />
            <Clip id="52312" at={6.0} rate={1.5} blend="screen" opacity={0.85} />
          </Shot>
          <Shot name="L roses on fire 2" from={256} to={266}>
            <Clip id="100898" at={7.0} rate={1.8} scale={1.7} x={-260} y={60} mirror />
            <Clip id="52312" at={10.0} rate={1.5} blend="screen" opacity={0.9} />
          </Shot>
          <Shot name="M rose burns" from={266} to={292}>
            <Clip id="171" at={12.9} rate={1.8} scale={1.2} y={-60} />
            <Clip id="3759" at={8.0} rate={1.5} blend="screen" scale={1.25} y={140} />
            <Clip id="4426" at={2.0} blend="screen" opacity={0.8} />
          </Shot>
          <Shot name="N flame masses" from={292} to={302}><Clip id="52304" at={0.8} rate={1.5} /></Shot>
          <Shot name="N flame masses 2" from={302} to={313}><Clip id="52312" at={12.0} rate={1.5} mirror scale={1.3} /></Shot>
          <Shot name="O watcher returns" from={313} to={325}><Clip id="1038" at={20.4} rate={1.5} scale={1.15} /></Shot>
          <Shot name="O watcher returns 2" from={325} to={337}><Clip id="1038" at={17.5} rate={1.5} scale={1.35} mirror /></Shot>
          <Shot name="P smoke" from={337} to={361}><Clip id="50951" at={0.2} rate={2.0} /></Shot>
          <Shot name="P seam birth" from={361} to={362}><Clip id="50951" at={1.867} rate={2.0} /></Shot>
        </AbsoluteFill>
        {birth && <AbsoluteFill style={{ backgroundColor: "#F7E6EA", mixBlendMode: "multiply" }} />}
        {impact && <AbsoluteFill style={{ backgroundColor: "#E31E3C", mixBlendMode: "multiply" }} />}
      </Camera>

      {/* light sources that charge the flashes */}
      {d >= 5 && d < 9 && <Point size={20 + (d - 5) * 28} power={0.5 + (d - 5) * 0.17} />}
      {d >= 98 && d < 102 && <Point size={24 + (d - 98) * 30} power={0.55 + (d - 98) * 0.15} x={0.56} y={0.4} />}
      {d >= 143 && d < 148 && <Point size={14 + (d - 143) * 18} power={0.5 + (d - 143) * 0.12} />}
      {d >= 192 && d < 196 && <Point size={20 + (d - 192) * 26} power={0.5 + (d - 192) * 0.15} x={0.5} y={0.28} />}
      {d >= 353 && d < 357 && <Point size={20 + (d - 353) * 28} power={0.5 + (d - 353) * 0.15} />}

      {flash && <FlashLayer f={flash} />}
      {d === 148 && <AbsoluteFill style={{ backgroundColor: "black" }} />}

      {/* texture: vignette (not on flashes — a full-frame white must stay full) + grain */}
      {!flash && !birth && (
        <AbsoluteFill
          style={{ background: "radial-gradient(ellipse at center, rgba(0,0,0,0) 58%, rgba(0,0,0,0.6) 100%)" }}
        />
      )}
      <Img
        src={staticFile(`fx/grain${Math.floor(random(`g-${d}`) * 6)}.png`)}
        style={{ position: "absolute", width: W, height: H, mixBlendMode: "overlay", opacity: 0.18 }}
      />
    </AbsoluteFill>
  );
};

export const RoseAsh: React.FC = () => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill style={{ backgroundColor: "black" }}>
      <Freeze frame={drawingOf(f)}>
        <Edit />
      </Freeze>
      <Audio src={staticFile("audio/track.m4a")} />
    </AbsoluteFill>
  );
};
