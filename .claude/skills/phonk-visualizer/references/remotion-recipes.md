# Рецепты в Remotion

Проект лежит в `video/` (Remotion 4.0.534). WebGL рендерится через SwiftShader (`swangle`, задано в `remotion.config.ts`). API сверены с пакетами 4.0.534. Если версия изменилась, проверь props через MCP `remotion-documentation` или прочитай `node_modules/@remotion/*/dist/*.d.ts`.

Пакеты: `npx remotion add @remotion/effects @remotion/motion-blur @remotion/transitions @remotion/media @remotion/three @remotion/noise`.

## Карта «приём из After Effects → Remotion»

| Приём (AE) | Remotion |
|---|---|
| Posterize Time («на двойках») | `<Freeze frame={held}>` вокруг сцены, где `held` — рамка рисунка (см. ниже) |
| Deep Glow | `glow({radius, intensity, threshold, color: '#E31E3C'})`, слоями 2–3 радиуса |
| Radial / zoom blur | `zoomBlur({amount, center, samples: 32+})`. Мало сэмплов дают полосы |
| CC Light Rays / направленный смаз | `lightTrail({direction, distance, intensity, decay, threshold, color})` |
| Motion blur на движении | `<CameraMotionBlur shutterAngle={180..360} samples={6..10}>` или `<Trail>` |
| Turbulent Displace | `noiseDisplacement({center, radius, strength, grainSize, seed, biasDirection, biasAmount})` |
| Tint / Colorama → duotone | `duotone({darkColor:'#0D050F', lightColor:'#E31E3C', threshold})` или `levels` + `tint` |
| Levels / Exposure (скачок экспозиции на рисунок) | `levels({blackPoint, whitePoint, gamma})`, `exposure({stops})` |
| Рваные края масс | `roughenEdges({amount, border, scale, seed})` |
| Пиксельный смаз | `pixelate({blockSize})` по маске, `pixelDissolve`, `linearProgressivePixelate` |
| Хроматическая аберрация на ударе | `chromaticAberration({amount, angle})` |
| Световые блики / лучи | `starburst`, `lightLeak({seed, hueShift, progress})` |
| Виньетка | `vignette` |
| 3D-частицы, осколки | `@remotion/three` (`<ThreeCanvas>`, всё анимируется только через `useCurrentFrame()`) |
| Шум для органики | `@remotion/noise` (`noise2D/3D`), детерминирован по seed |
| Склейки и оверлеи | `<TransitionSeries>` + `<TransitionSeries.Overlay>` (вспышки поверх склейки, без укорачивания) |

Эффекты подключаются через проп `effects` у `<Img>`, `<Solid>`, `<Video>` (из `@remotion/media`) и `<HtmlInCanvas>`. Для `<AbsoluteFill>` и `<Sequence>` контент нужно обернуть в `<HtmlInCanvas>`.

## Рваный fps, который читается как стиль

```tsx
import {Freeze, useCurrentFrame} from 'remotion';

// holds: [{from, to, hold}] — отрезки с шагом удержания 1/2/3
const heldFrame = (f: number, holds: {from: number; to: number; hold: number}[]) => {
  const h = holds.find((s) => f >= s.from && f < s.to);
  return h ? h.from + Math.floor((f - h.from) / h.hold) * h.hold : f;
};

export const Stepped: React.FC<{holds: any[]; children: React.ReactNode}> = ({holds, children}) => {
  const f = useCurrentFrame();
  return <Freeze frame={heldFrame(f, holds)}>{children}</Freeze>;
};
```

Обязательные условия, иначе получится лаг:
- **Каждый рисунок — новая поза.** Анимацию строй ключами-позами (`interpolate` по ключам рисунков), а не плавной кривой, которую потом проредили.
- **Скачок экспозиции на каждом рисунке:** `exposure({stops: pose.exposure})`, где значения чередуются вроде −0.4 / +0.3 / −0.8 / +0.6.
- **Смаз внутри кадра:** `<CameraMotionBlur>` с большим shutterAngle на быстрых позах плюс `lightTrail` по направлению движения.
- **Камера прыгает позами** (смена масштаба и сдвиг на ударе), а не плывёт. Если нужен плавный наезд, этот слой вынеси из `<Freeze>` и рендери «на единицах».

## Материал: живое видео через стилизацию

```tsx
import {Video} from '@remotion/media';
import {duotone} from '@remotion/effects/duotone';
import {levels} from '@remotion/effects/levels';
import {glow} from '@remotion/effects/glow';
import {staticFile} from 'remotion';

<Video
  src={staticFile('footage/silhouette.mp4')}
  muted
  effects={[
    levels({blackPoint: 0.18, whitePoint: 0.85, gamma: 0.9}),
    duotone({darkColor: '#0D050F', lightColor: '#E31E3C', threshold: 0.45}),
    glow({radius: 28, intensity: 0.8, threshold: 0.55, color: '#E31E3C'}),
  ]}
/>
```

- Футажи клади в `video/public/footage/`. Источник и лицензию записывай в `video/ASSETS.md`.
- Подходящий материал: силуэты людей (танец, медленные движения, руки вверх), дым и чернила на чёрном, огонь на чёрном, частицы и пыль на чёрном.

## Переход «сцена рождается во вспышке»

```tsx
<TransitionSeries>
  <TransitionSeries.Sequence durationInFrames={28} premountFor={fps}><SceneA /></TransitionSeries.Sequence>
  <TransitionSeries.Overlay durationInFrames={8} premountFor={fps}>
    {/* 2 кадра: точка-вспышка; 2 кадра: белое с призраком A; 1 кадр: из белого проступает зародыш B */}
    <FlashBirth />
  </TransitionSeries.Overlay>
  <TransitionSeries.Sequence durationInFrames={40} premountFor={fps}><SceneB /></TransitionSeries.Sequence>
</TransitionSeries>
```

В `FlashBirth` нужно смешивать по кадрам: призрак A в белом, затем зародыш B в белом. Делать это ручными позами, а не одной кривой прозрачности.

## Пролёт сквозь форму

1. 3–4 кадра разгона: `zoomBlur` нарастает, масштаб растёт геометрически (×1.15 на рисунок).
2. 2 кадра: форма за краем кадра, всё в розово-белом расфокусе (`blur` + `exposure`).
3. 1 кадр: на переднем плане пролетает крупный размытый обломок.
4. Фокус: новая сцена с виньеткой.

## Проверка в процессе

- Ключевые кадры рендери так: `npx remotion render <id> out/frames --frames=0,30,90 --image-format=png`.
- Перед каждым полным рендером собирай полоски переходов и сравнивай с референсом рядом (`scripts/critic_packet.py`).
- Полный рендер: `npx remotion render <id> out/final.mp4 --codec=h264 --crf=18`. Потом, если файл больше 30 МБ, пережми копию для отправки до 16 Мбит/с.
