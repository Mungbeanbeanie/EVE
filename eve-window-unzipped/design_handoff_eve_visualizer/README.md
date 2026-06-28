# Handoff: Eve — Voice Visualizer Window

## Overview
Eve is an on-device AI personal-assistant window. The centerpiece is a **fluid neural-network orb** voice visualizer: a rotating 3D particle sphere whose nodes are wired together by "neural" connections, with bright signal pulses firing along the edges, a glowing reactive core, orbital rings, and HUD arcs. The orb reacts to Eve's conversational state (idle / listening / thinking / speaking). The window is a small, calm, minimal, **translucent glass panel** with macOS chrome and two input modes: **push-to-talk** and **always-on**.

The look is deliberately futuristic/"JARVIS"-like — warm amber particles on a dark translucent background (see `reference/inspiration.png`).

## About the Design Files
The files in `reference/` are a **design reference created in HTML** — a working prototype showing the intended look and behavior. They are **not production code to copy directly**. `reference/Eve.dc.html` runs inside a proprietary preview runtime (it relies on a `support.js` harness and a `renderVals()` template-binding layer that you should **not** port).

Your task is to **recreate this design in the target codebase's environment** (React, Electron, Tauri, SwiftUI/Canvas, plain web component, etc.) using its established patterns. If no environment exists yet, choose what fits an on-device desktop assistant — an **Electron or Tauri app** with a web frontend is the natural fit, since the visualizer is HTML5 Canvas.

**To save you the extraction work, `src/eve-orb.js` is already a clean, dependency-free version of the renderer** — plain ES-module class, no preview runtime, with a real public API and optional microphone reactivity. Start there; it is the genuinely portable core. Use the HTML prototype only as the visual + interaction spec for the surrounding window chrome and controls.

## Fidelity
**High-fidelity.** Colors, typography, spacing, motion, and the visualizer algorithm are final. Recreate the UI to match. The one intentional gap: the prototype animates the orb's "loudness" with a **synthetic envelope** because it has no audio. In the real app you should drive it from **actual microphone amplitude** (and from your real assistant state machine). `src/eve-orb.js` already supports this — see *Integration*.

---

## Files in this bundle
- `src/eve-orb.js` — **clean vanilla-JS renderer.** Zero dependencies. This is what you build on.
- `eve.config.json` — commented (JSONC) tuning file: palettes, particle count, depth-of-field, rotation, per-state energy, demo timings. The renderer accepts this object as its constructor config.
- `reference/Eve.dc.html` — the full HTML prototype (window chrome + controls + orb). Visual/interaction spec only.
- `reference/inspiration.png` — the visual north star the design was built to match.

---

## The Visualizer (`src/eve-orb.js`)

### What it draws, layer by layer (additive `'lighter'` blending throughout)
1. **Particle sphere** — `particleCount` nodes placed evenly on a unit sphere via the **Fibonacci/golden-angle distribution**, rotated each frame (`rotation.yaw` spin + gently wobbling `rotation.pitch`), and projected to 2D with a simple perspective divide (`fov = 2.7`). Each node's **depth** `d` runs 0 (back) → 1 (front).
2. **Neural edges** — each node connects to its `neighbors` (default 3) nearest nodes. Edge opacity scales with average depth² so the front of the mesh reads brighter.
3. **Depth-of-field** — far nodes (`d < depthOfField.nearPlane`) are drawn with a canvas `shadowBlur` bokeh up to `maxBlur` px and dimmed by `farDim`; near nodes stay crisp. Nodes are painted back-to-front so the blur composites correctly. This is the "subtle 3D" depth cue.
4. **Signal pulses** — small bright dots travel along random edges; count = `states[state].pulses`, intensity = `states[state].brightness`, speed scales with live energy.
5. **Core glow** — a radial gradient at the center that swells with energy.
6. **Orbital rings + HUD arcs** — three tilted great-circle rings and two rotating arc sets in the `ring` color.

### Energy / reactivity
A single `energy()` value (0..1) modulates radius, brightness, breathing, pulse speed. Source priority:
1. A live **Web Audio `AnalyserNode`** (call `orb.attachMic(analyser)`), or
2. A manually fed level (`orb.setLevel(0..1)` each frame), or
3. The built-in **synthetic per-state envelope** (default, used by the prototype).

### Public API
```js
const orb = new EveOrb(canvasEl, config);  // config = parsed eve.config.json (optional; deep-merged over defaults)
orb.setState('idle' | 'listening' | 'thinking' | 'speaking');
orb.setAccent('amber' | 'cyan' | 'violet' | 'mono');  // any key in config.palettes
orb.attachMic(analyserNode);   // drive energy from real audio
orb.setLevel(0..1);            // or feed your own level
orb.resize();                  // after the canvas CSS box changes
orb.destroy();                 // stop the RAF loop + detach
```
The class owns its own `requestAnimationFrame` loop and handles `devicePixelRatio`. It expects the canvas to be sized via CSS (it reads `getBoundingClientRect()` and sets the backing store).

---

## Window & Controls (spec — recreate from `reference/Eve.dc.html`)

### Window shell
- **Outer ambience (the "desktop behind the window"):** dark radial background `radial-gradient(120% 120% at 30% 10%, #15171f, #0c0d12 55%, #07070b)` with two large blurred drifting color blobs (accent-tinted, `filter: blur(80–90px)`, ~17–21s `ease-in-out` drift loops) and a faint masked grid (46px cells, `rgba(255,255,255,0.025)`, radial-masked to fade at edges). In the real app this sits behind/around the panel; the panel itself is what matters.
- **Glass panel:** width **372px**, `border-radius: 22px`, `background: rgba(15,17,24,0.46)`, `border: 1px solid rgba(255,255,255,0.10)`, `backdrop-filter: blur(26px) saturate(150%)` (this is what makes the desktop show through — keep it), layered inset highlights + a soft inner accent halo, and a big drop shadow `0 40px 90px rgba(0,0,0,0.55)`.
- **Title bar:** 40px tall, bottom border `1px solid rgba(255,255,255,0.07)`. Left: three macOS traffic-light dots (`#ff5f57`, `#febc2e`, `#28c840`), 11px circles, 8px gap. Center: `EVE`, 12px, weight 600, letter-spacing 3px, `rgba(238,241,246,0.85)`.
- **Canvas stage:** full panel width × **300px** tall. The orb renders here. A status line overlays the bottom center: a 7px glowing dot + uppercase label (11px, weight 600, letter-spacing 2.5px), e.g. `READY`, `LISTENING`, `THINKING`, `SPEAKING`, `MUTED`.

### Typography
**Space Grotesk** (Google Fonts; weights 400/500/600/700) for all UI text. Status/labels are uppercase with wide letter-spacing.

### Controls (below the canvas, 14–16px padding)
1. **Mode segmented control** — a pill track (`rgba(255,255,255,0.05)` bg, `border-radius: 12px`, 3px padding) with two segments: **Push-to-talk** and **Always on**. The active segment gets `background: rgba(255,255,255,0.10)` and full-opacity text (`rgba(238,241,246,0.92)`); inactive text is `rgba(238,241,246,0.42)`. Segment labels 12px, weight 600.
2. **Push-to-talk button** (visible in PTT mode) — full-width, `border-radius: 13px`, padding 13px, `rgba(255,255,255,0.05)` bg with `1px solid rgba(255,255,255,0.10)` border. Contains a small dot + label "Hold to talk". **While held:** background becomes the solid accent (`palettes[accent].hex`), text goes dark (`#0b0c10`), label changes to "Listening… release to send", and a glow shadow appears (`0 8px 26px rgba(edge,0.5)`). Behavior: `mousedown` → state `listening`; `mouseup`/`mouseleave` → state `thinking`, then after `timings.thinkingDelayMs` → `speaking`, then after `timings.replyDurationMs` → back to `idle`.
3. **Always-on controls** (visible in Always-on mode) — two side-by-side buttons:
   - **Mute mic / Unmute** — toggles muted. When muted: solid `#ff6b6b` bg, dark text, status shows `MUTED`, orb goes idle.
   - **Demo reply** — solid accent bg, dark text; triggers thinking → speaking → listening (a canned reply for preview).
4. **Preview row** — a small `PREVIEW` label (9.5px, letter-spacing 1.2px, `rgba(238,241,246,0.35)`) + four tiny scrub buttons **Idle / Listen / Think / Speak** that force the orb into each state. *This row is a demo affordance for the prototype — drop it in production and drive state from the real assistant.*

All buttons: `transition: background .16s, color .16s, transform .1s`; `:active { transform: scale(0.975) }`.

> **Important pattern note:** in the prototype, every button that changes appearance puts its background/border/color on an **inner `<span>`** while the `<button>` stays transparent. That's a workaround for the preview runtime and is **not needed** in a normal React/Vue/etc. app — style the button directly there.

---

## Interactions & Behavior
- **Mode switch** resets state: PTT → `idle`; Always-on → `listening`.
- **PTT press/hold** → `listening`; **release** → `thinking` → (`thinkingDelayMs`) → `speaking` → (`replyDurationMs`) → `idle`.
- **Mute** (always-on) → `idle` + `MUTED`; **unmute** → `listening`.
- **State → orb:** each state sets pulse count + brightness (`eve.config.json → states`) and changes the synthetic energy curve (replace with mic amplitude in production).
- **Status dot colors:** idle `rgba(238,241,246,0.42)` (or `#ff6b6b` when muted), listening `#3ad29b`, thinking `#ffc44d`, speaking = accent hex.
- Animations are continuous `requestAnimationFrame`; there are no CSS keyframe transitions on the orb itself (it's all canvas). The drifting background blobs use CSS `@keyframes`.

## State Management
Minimal. Two pieces of UI state plus the orb's visual state:
- `mode`: `'ptt' | 'always'`
- `muted`: boolean (always-on only)
- `anim`: `'idle' | 'listening' | 'thinking' | 'speaking'` — **the only thing the orb needs.** Call `orb.setState(anim)` whenever it changes.
- In production, `anim` is derived from your real pipeline: mic VAD / wake-word → `listening`; LLM request in flight → `thinking`; TTS playing → `speaking`; otherwise `idle`.

## Integration (wiring real audio)
```js
// 1. Render the orb
import { EveOrb } from './eve-orb.js';
import config from './eve.config.json' assert { type: 'json' }; // or fetch+parse (strip // comments)
const orb = new EveOrb(document.querySelector('#eve-canvas'), config);

// 2. Drive energy from the mic while listening/speaking
const ac = new AudioContext();
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const src = ac.createMediaStreamSource(stream);
const analyser = ac.createAnalyser();
analyser.fftSize = 256;
src.connect(analyser);
orb.attachMic(analyser);        // orb now pulses to real loudness

// For Eve's OWN voice (TTS) reactivity, connect the TTS audio element through
// a second analyser and swap which one is attached when she starts speaking.

// 3. Drive state from your assistant pipeline
orb.setState('listening');      // wake word detected
orb.setState('thinking');       // request sent to model
orb.setState('speaking');       // TTS playback started
orb.setState('idle');           // done
```
- `eve.config.json` is **JSONC** (has `//` comments). If your bundler imports JSON strictly, either strip comments at build time or convert it to plain `.json`. The prototype strips comments at runtime before `JSON.parse`.
- If you don't want mic access, omit `attachMic` and the built-in per-state envelope animates the orb on its own.

## Design Tokens
**Palettes** (`edge` = lines, `node` = dots, `core` = pulse heads/center, `ring` = rings/HUD, `hex` = solid accent for buttons/status):
- amber — edge `#ff8c1a` · node `#ffcf7a` · core `#ffe7b8` · ring `#ffb347` · hex `#ffb65e`
- cyan — edge `#1fb0e0` · node `#8fe9ff` · core `#d8fbff` · ring `#52d4f0` · hex `#6fdcf5`
- violet — edge `#8a6cff` · node `#c4b3ff` · core `#ece4ff` · ring `#a98cff` · hex `#b49cff`
- mono — edge `#9fb0c8` · node `#e3ebf5` · core `#ffffff` · ring `#b8c6da` · hex `#cdd8e6`

**Surfaces / text (dark glass theme):**
- Panel bg `rgba(15,17,24,0.46)` · panel border `rgba(255,255,255,0.10)` · hairline `rgba(255,255,255,0.07)`
- Control surface `rgba(255,255,255,0.05)` · active surface `rgba(255,255,255,0.10)`
- Text primary `rgba(238,241,246,0.92)` · text muted `rgba(238,241,246,0.42)` · text faint `rgba(238,241,246,0.35)`
- Dark-on-accent text `#0b0c10` · destructive (mute) `#ff6b6b` · listening `#3ad29b` · thinking `#ffc44d`

**Geometry:** panel radius 22px · button radius 13px · segment radius 9–12px · pill chips radius 8px.
**Backdrop blur:** panel `blur(26px) saturate(150%)`.
**Type:** Space Grotesk 400/500/600/700.

**Tunables (`eve.config.json`):** `particleCount` (80–240, default 150) · `neighbors` (3) · `depthOfField {nearPlane 0.82, maxBlur 7, farDim 0.45}` · `rotation {yaw 0.16, pitchBase 0.34, pitchWobble 0.14}` · `states.*.{pulses,brightness}` · `timings {thinkingDelayMs 650, replyDurationMs 3200}`.

## Assets
- `reference/inspiration.png` — visual reference only; not used at runtime.
- Fonts: **Space Grotesk** via Google Fonts. No icon library required (traffic-light dots and status dots are plain CSS circles; the orb is pure canvas).
- No images ship in the running component.

## Suggested build order
1. Drop `src/eve-orb.js` into the project; render it on a `<canvas>` sized by CSS; confirm the sphere animates.
2. Feed `eve.config.json`; verify palettes / particle count / DoF respond.
3. Build the glass panel + title bar + status line around it.
4. Build the segmented mode control and the PTT / always-on buttons; wire them to `orb.setState(...)`.
5. Replace the synthetic envelope with a real mic `AnalyserNode` and connect `anim` to your assistant's actual listening/thinking/speaking events.
6. Drop the `PREVIEW` scrub row.
