import React from "react";
import {
  AbsoluteFill,
  Freeze,
  Img,
  random,
  Sequence,
  staticFile,
  useCurrentFrame,
} from "remotion";
import { Audio, Video } from "@remotion/media";
import { zoomBlur } from "@remotion/effects/zoom-blur";
import { chromaticAberration } from "@remotion/effects/chromatic-aberration";
import { lightTrail } from "@remotion/effects/light-trail";

// "Rose / Ash" — see video/PLAN.md for the shot table.
export const DURATION = 362;
const W = 1920;
const H = 1080;

// Kicks of the user's track (frames @30fps), from analyze_audio.py.
const KICKS = [9, 32, 56, 79, 102, 149, 173, 196, 220, 243, 266, 290, 313, 337, 360];
const SUBHITS = [21, 118, 253, 275, 325];

// Posterize-time: every drawing is a new pose. Ones on hits/velocity, threes in the break.
const SEGS: { from: number; to: number; hold: number }[] = [
  { from: 0, to: 9, hold: 2 },
  { from: 9, to: 13, hold: 1 },
  { from: 13, to: 21, hold: 2 },
  { from: 21, to: 32, hold: 2 },
  { from: 32, to: 56, hold: 2 },
  { from: 56, to: 79, hold: 2 },
  { from: 79, to: 90, hold: 2 },
  { from: 90, to: 102, hold: 2 },
  { from: 102, to: 106, hold: 1 },
  { from: 106, to: 122, hold: 2 },
  { from: 122, to: 134, hold: 1 },
  { from: 134, to: 146, hold: 3 },
  { from: 146, to: 149, hold: 1 },
  { from: 149, to: 157, hold: 1 },
  { from: 157, to: 173, hold: 2 },
  { from: 173, to: 196, hold: 2 },
  { from: 196, to: 200, hold: 1 },
  { from: 200, to: 220, hold: 2 },
  { from: 220, to: 243, hold: 2 },
  { from: 243, to: 247, hold: 1 },
  { from: 247, to: 266, hold: 2 },
  { from: 266, to: 290, hold: 2 },
  { from: 290, to: 294, hold: 1 },
  { from: 294, to: 313, hold: 2 },
  { from: 313, to: 337, hold: 2 },
  { from: 337, to: 360, hold: 2 },
  { from: 360, to: 362, hold: 1 },
];

export const drawingOf = (f: number) => {
  const s = SEGS.find((x) => f >= x.from && f < x.to) ?? SEGS[SEGS.length - 1];
  return s.from + Math.floor((f - s.from) / s.hold) * s.hold;
};

// strength of the latest hit at drawing d (1 on the hit, halves every drawing)
const hitAt = (d: number) => {
  let v = 0;
  for (const k of KICKS) if (d >= k && d - k < 10) v = Math.max(v, Math.pow(0.5, (d - k) / 2));
  for (const k of SUBHITS) if (d >= k && d - k < 6) v = Math.max(v, 0.6 * Math.pow(0.5, (d - k) / 2));
  return v;
};

// ---------------------------------------------------------------- layers

type ClipProps = {
  id: string;
  at: number; // source seconds at the start of the sequence
  from: number; // absolute start frame of the enclosing Sequence (for hit lookup)
  rate?: number;
  mirror?: boolean;
  scale?: number;
  x?: number;
  y?: number;
  blend?: React.CSSProperties["mixBlendMode"];
  opacity?: number;
  trail?: { direction: number; distance: number; color: string };
  zoom?: (abs: number) => number; // extra zoom-blur amount
};

const Clip: React.FC<ClipProps> = (p) => {
  const local = useCurrentFrame();
  const abs = local + p.from;
  const hit = hitAt(abs);
  const zb = p.zoom ? p.zoom(abs) : 0;
  const fx = [];
  if (p.trail) fx.push(lightTrail({ ...p.trail, intensity: 0.9, decay: 0.92, threshold: 0.45, samples: 28 }));
  if (zb > 0.5) fx.push(zoomBlur({ amount: zb, center: [0.5, 0.5], samples: 64 }));
  if (hit > 0.2) fx.push(chromaticAberration({ amount: 3 + 12 * hit, angle: 0 }));
  const s = p.scale ?? 1;
  return (
    <AbsoluteFill
      style={{
        mixBlendMode: p.blend ?? "normal",
        opacity: p.opacity ?? 1,
        transform: `translate(${p.x ?? 0}px, ${p.y ?? 0}px) scale(${(p.mirror ? -1 : 1) * s}, ${s})`,
      }}
    >
      <Video
        src={staticFile(`footage/graded/${p.id}.mp4`)}
        trimBefore={Math.round(p.at * 30)}
        playbackRate={p.rate ?? 1}
        muted
        objectFit="cover"
        effects={fx}
        style={{ width: W, height: H }}
      />
    </AbsoluteFill>
  );
};

const Shot: React.FC<{ from: number; to: number; children: (from: number) => React.ReactNode; name: string }> = ({
  from,
  to,
  children,
  name,
}) => (
  <Sequence from={from} durationInFrames={to - from} name={name} layout="none">
    <AbsoluteFill>{children(from)}</AbsoluteFill>
  </Sequence>
);

const Point: React.FC<{ size: number; power: number }> = ({ size, power }) => (
  <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", mixBlendMode: "screen" }}>
    <div
      style={{
        width: size * 4,
        height: size * 4,
        borderRadius: "50%",
        background: `radial-gradient(circle, rgba(255,246,244,${Math.min(1, power)}) 0%, rgba(255,240,236,${
          0.9 * power
        }) 12%, rgba(227,30,60,${0.55 * power}) 30%, rgba(120,8,24,${0.25 * power}) 55%, rgba(0,0,0,0) 75%)`,
      }}
    />
  </AbsoluteFill>
);

// ---------------------------------------------------------------- camera + exposure (per drawing)

const Camera: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const d = useCurrentFrame(); // inside <Freeze>: the drawing frame
  const hit = hitAt(d);
  // new pose every drawing: jump, not glide
  const r = (k: string) => random(`${k}-${d}`) * 2 - 1;
  let scale = 1.07 + 0.045 * r("s") + 0.1 * hit;
  let tx = 34 * r("x") + 40 * hit * r("hx");
  let ty = 20 * r("y") + 30 * hit * r("hy");
  let rot = 0.9 * r("r") + 1.6 * hit * r("hr");
  // velocity dive into the heart of the rose (122–133)
  if (d >= 122 && d < 134) {
    const k = (d - 122) / 11;
    scale = 1.06 * Math.pow(3.2, k * k);
    tx *= 0.3;
    ty *= 0.3;
    rot = 6 * k * k;
  }
  // exposure strobe: every drawing gets its own exposure; hits pop
  let bright = 1 + 0.32 * r("e") + 0.45 * hit;
  if (d >= 134 && d < 149) bright = 0.8 + 0.2 * r("e");
  // born in light: the incoming scene is overexposed, then settles (black stays black)
  const BURN: Record<number, number> = { 10: 2.8, 11: 1.8, 12: 1.3, 103: 2.6, 104: 1.6, 150: 3.0, 151: 2.0, 152: 1.4,
    197: 2.7, 198: 1.7, 199: 1.25, 244: 2.7, 245: 1.7, 246: 1.25 };
  const burn = BURN[d] ? (BURN[d] - 1) / 2 : 0; // 0..1
  return (
    <AbsoluteFill
      style={{
        transform: `translate(${tx}px, ${ty}px) rotate(${rot}deg) scale(${scale})`,
        filter: `brightness(${bright.toFixed(3)}) contrast(${(1.08 + 0.1 * hit).toFixed(3)})`,
      }}
    >
      {children}
      {burn > 0 && (
        // red burn (color-dodge): red channel blows out, blue/green barely move -> crimson, never pink
        <AbsoluteFill
          style={{
            mixBlendMode: "color-dodge",
            backgroundColor: `rgb(${Math.round(255 * (0.25 + 0.5 * burn))}, ${Math.round(255 * 0.22 * burn)}, ${Math.round(
              255 * 0.2 * burn,
            )})`,
          }}
        />
      )}
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------- the edit (rendered on drawings)

const Edit: React.FC = () => {
  const d = useCurrentFrame();
  const dive = (abs: number) => (abs >= 122 && abs < 134 ? 10 + 14 * (abs - 122) : 0);
  const dropBlur = (abs: number) => (abs >= 150 && abs < 153 ? [40, 20, 8][abs - 150] : 0);

  // flash births: white over the outgoing scene, the incoming scene is born inside it
  const births: Record<number, number> = { 9: 0.92, 102: 0.88, 149: 1.0, 196: 0.9, 243: 0.9 };
  const white = births[d] ?? 0;
  const impact = d === 290 || d === 291;
  const black = d === 148;

  return (
    <AbsoluteFill style={{ backgroundColor: "black" }}>
      <Camera>
        <AbsoluteFill style={impact ? { filter: "grayscale(1) contrast(5) invert(1)" } : undefined}>
          {/* A 0–9: smoke breathing — continues the end of the loop */}
          <Shot name="A smoke" from={0} to={10}>{(f) => <Clip id="50951" at={1.867} rate={2.0} from={f} />}</Shot>
          {/* B 10–31: the rose, born in the flash */}
          <Shot name="B rose" from={10} to={21}>{(f) => <Clip id="171" at={2.0} rate={2.0} from={f} scale={1.15} />}</Shot>
          <Shot name="B rose 2" from={21} to={32}>{(f) => <Clip id="171" at={4.6} rate={2.0} from={f} scale={1.35} x={160} mirror />}</Shot>
          {/* C 32–55: closer, other side; vocal line → light trail up */}
          <Shot name="C rose close" from={32} to={56}>
            {(f) => (
              <><Clip id="171" at={7.0} rate={2.0} from={f} scale={1.4} x={-120} y={40} mirror />
                <Clip id="3465" at={2.0} from={f} blend="screen" /></>
            )}
          </Shot>
          {/* D 56–78: hands reach (crimson) */}
          <Shot name="D hands" from={56} to={79}>{(f) => <Clip id="40938" at={4.5} rate={1.6} from={f} scale={1.1} />}</Shot>
          {/* E 79–101: the watcher in the beam; pose jump at 90 */}
          <Shot name="E figure" from={79} to={90}>{(f) => <Clip id="1038" at={1.2} rate={1.4} from={f} scale={1.2} />}</Shot>
          <Shot name="E figure 2" from={90} to={102}>{(f) => <Clip id="1038" at={10.8} rate={1.4} from={f} scale={1.2} x={110} y={-40} />}</Shot>
          {/* F 102–133: macro rose catches fire; dive into the heart */}
          <Shot name="F macro" from={102} to={134}>
            {(f) => (
              <>
                <Clip id="100917" at={1.0} rate={1.6} from={f} zoom={dive} />
                <Clip id="3759" at={4.0} from={f} blend="screen" scale={1.1} y={60} />
              </>
            )}
          </Shot>
          {/* G 134–147: the inhale — almost black, embers */}
          <Shot name="G break" from={134} to={148}>{(f) => <Clip id="4426" at={6.0} from={f} opacity={0.5} />}</Shot>
          {/* H 149–172: drop — wet roses + fire burst */}
          <Shot name="H drop" from={149} to={173}>
            {(f) => (
              <>
                <Clip id="100899" at={2.0} rate={1.4} from={f} zoom={dropBlur} scale={1.1} />
                <Clip id="52304" at={0.6} from={f} blend="screen" zoom={dropBlur} />
              </>
            )}
          </Shot>
          {/* I 173–195: blood ink blooms */}
          <Shot name="I ink" from={173} to={185}>{(f) => <Clip id="41999" at={2.2} rate={2.5} from={f} mirror scale={1.1} />}</Shot>
          <Shot name="I ink 2" from={185} to={197}>{(f) => <Clip id="41999" at={9.0} rate={2.5} from={f} scale={1.35} y={-60} />}</Shot>
          {/* J 197–242: dancer in smoke; jump at 220 */}
          <Shot name="J dancer" from={197} to={220}>{(f) => <Clip id="33899" at={8.4} rate={1.3} from={f} scale={1.15} />}</Shot>
          <Shot name="J dancer 2" from={220} to={244}>{(f) => <Clip id="33899" at={14.0} rate={1.3} from={f} scale={1.4} y={80} mirror />}</Shot>
          {/* K 244–289: cold hands sing the vocal line, crimson trails */}
          <Shot name="K cold hands" from={244} to={266}>
            {(f) => (
              <>
                <Clip id="3465" at={4.0} from={f} blend="screen" scale={1.2} />
                <Clip id="40938c" at={7.1} rate={1.6} from={f} blend="screen" />
              </>
            )}
          </Shot>
          <Shot name="K cold hands 2" from={266} to={290}>
            {(f) => (
              <>
                <Clip id="4426" at={1.5} from={f} blend="screen" />
                <Clip id="40938c" at={9.6} rate={1.6} from={f} mirror scale={1.25} blend="screen" />
              </>
            )}
          </Shot>
          {/* L 290–312: roses in fire */}
          <Shot name="L roses fire" from={290} to={313}>
            {(f) => (
              <>
                <Clip id="100898" at={3.0} rate={1.8} from={f} />
                <Clip id="52312" at={6.0} rate={1.5} from={f} blend="screen" opacity={0.85} />
              </>
            )}
          </Shot>
          {/* M 313–336: the rose burns */}
          <Shot name="M rose burns" from={313} to={337}>
            {(f) => (
              <>
                <Clip id="171" at={12.9} rate={1.8} from={f} scale={1.2} y={-60} />
                <Clip id="3759" at={8.0} rate={1.5} from={f} blend="screen" scale={1.25} y={140} />
                <Clip id="4426" at={2.0} from={f} blend="screen" opacity={0.8} />
              </>
            )}
          </Shot>
          {/* N 337–361: smoke — runs into frame 0 */}
          <Shot name="N smoke" from={337} to={362}>{(f) => <Clip id="50951" at={0.2} rate={2.0} from={f} />}</Shot>
        </AbsoluteFill>
        {impact && <AbsoluteFill style={{ backgroundColor: "#E31E3C", mixBlendMode: "multiply" }} />}
      </Camera>

      {/* charging points before the flashes */}
      {d >= 5 && d < 9 && <Point size={20 + (d - 5) * 26} power={0.5 + (d - 5) * 0.17} />}
      {d >= 140 && d < 148 && <Point size={10 + (d - 140) * 14} power={0.4 + (d - 140) * 0.08} />}
      {d >= 98 && d < 102 && <Point size={30 + (d - 98) * 30} power={0.5 + (d - 98) * 0.15} />}

      {white > 0 && (
        <AbsoluteFill
          style={{
            background: `radial-gradient(ellipse at center, rgba(255,247,245,${white}) 0%, rgba(255,236,232,${white}) 45%, rgba(240,200,205,${
              white * 0.92
            }) 100%)`,
          }}
        />
      )}
      {black && <AbsoluteFill style={{ backgroundColor: "black" }} />}

      {/* texture: vignette + grain (new grain every drawing) */}
      <AbsoluteFill
        style={{ background: "radial-gradient(ellipse at center, rgba(0,0,0,0) 52%, rgba(0,0,0,0.72) 100%)" }}
      />
      <Img
        src={staticFile(`fx/grain${Math.floor(random(`g-${d}`) * 6)}.png`)}
        style={{ position: "absolute", width: W, height: H, mixBlendMode: "overlay", opacity: 0.2 }}
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
