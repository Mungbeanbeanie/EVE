/*
 * EveOrb — fluid neural-network voice visualizer.
 * Zero dependencies. Vanilla Canvas 2D. Framework-agnostic.
 *
 * This is a clean extraction of the renderer from the HTML prototype
 * (reference/Eve.dc.html). Use it as the starting point for the real
 * component — it has no prototype-runtime glue, just a plain class.
 *
 * USAGE
 * -----
 *   import { EveOrb } from './eve-orb.js';        // or <script type="module">
 *   const orb = new EveOrb(canvasEl, config);     // config = parsed eve.config.json (optional)
 *   orb.setState('speaking');                      // 'idle' | 'listening' | 'thinking' | 'speaking'
 *   orb.setAccent('cyan');                         // key from config.palettes
 *
 *   // Optional: drive the energy from a real mic instead of the synthetic envelope.
 *   orb.attachMic(analyserNode);                   // a Web Audio AnalyserNode
 *   // ...or feed your own 0..1 level each frame:
 *   orb.setLevel(0.0 .. 1.0);
 *
 *   orb.destroy();                                 // stop RAF + listeners
 *
 * The class owns its own requestAnimationFrame loop and resizes to the
 * canvas's CSS box (handles devicePixelRatio). Call resize() if you
 * change the canvas size manually.
 */

const DEFAULTS = {
  defaultAccent: 'amber',
  palettes: {
    amber:  { edge: '#ff8c1a', node: '#ffcf7a', core: '#ffe7b8', ring: '#ffb347', hex: '#ffb65e' },
    cyan:   { edge: '#1fb0e0', node: '#8fe9ff', core: '#d8fbff', ring: '#52d4f0', hex: '#6fdcf5' },
    violet: { edge: '#8a6cff', node: '#c4b3ff', core: '#ece4ff', ring: '#a98cff', hex: '#b49cff' },
    mono:   { edge: '#9fb0c8', node: '#e3ebf5', core: '#ffffff', ring: '#b8c6da', hex: '#cdd8e6' },
  },
  particleCount: 150,
  neighbors: 3,
  depthOfField: { enabled: true, nearPlane: 0.82, maxBlur: 7, farDim: 0.45 },
  rotation: { yaw: 0.16, pitchBase: 0.34, pitchWobble: 0.14 },
  states: {
    idle:      { pulses: 12, brightness: 0.5 },
    listening: { pulses: 20, brightness: 0.8 },
    thinking:  { pulses: 16, brightness: 0.7 },
    speaking:  { pulses: 30, brightness: 1.0 },
  },
};

export class EveOrb {
  constructor(canvas, config = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.C = deepMerge(DEFAULTS, config);
    this.accent = this.C.defaultAccent;
    this.anim = 'idle';
    this._micLevel = null;   // when set (0..1), overrides synthetic envelope
    this._analyser = null;
    this._freq = null;

    this.resize();
    this.build(this.C.particleCount);
    this._t0 = performance.now();
    this._loop = this._loop.bind(this);
    this._raf = requestAnimationFrame(this._loop);
  }

  // ---- public API ----
  setState(name) { if (this.C.states[name]) this.anim = name; }
  setAccent(key) { if (this.C.palettes[key]) this.accent = key; }
  setLevel(v)    { this._micLevel = v == null ? null : Math.max(0, Math.min(1, v)); }

  attachMic(analyserNode) {
    this._analyser = analyserNode;
    this._freq = new Uint8Array(analyserNode.frequencyBinCount);
  }
  detachMic() { this._analyser = null; this._freq = null; this._micLevel = null; }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const r = this.canvas.getBoundingClientRect();
    this._w = r.width || 372;
    this._h = r.height || 300;
    this.canvas.width = this._w * dpr;
    this.canvas.height = this._h * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  destroy() { cancelAnimationFrame(this._raf); this.detachMic(); }

  // ---- geometry build (Fibonacci sphere + nearest-neighbour neural edges) ----
  build(N) {
    N = Math.max(60, Math.min(260, N | 0));
    const nodes = [];
    const gold = Math.PI * (3 - Math.sqrt(5));        // golden angle → even sphere distribution
    for (let i = 0; i < N; i++) {
      const y = 1 - 2 * (i + 0.5) / N;
      const rr = Math.sqrt(Math.max(0, 1 - y * y));
      const th = gold * i;
      nodes.push({ x: Math.cos(th) * rr, y, z: Math.sin(th) * rr, ph: Math.random() * 6.28 });
    }
    const NB = this.C.neighbors || 3;
    const seen = new Set(), edges = [];
    for (let i = 0; i < N; i++) {
      const d = [];
      for (let j = 0; j < N; j++) if (j !== i) {
        const a = nodes[i], b = nodes[j];
        const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        d.push([dx * dx + dy * dy + dz * dz, j]);
      }
      d.sort((p, q) => p[0] - q[0]);
      for (let k = 0; k < NB; k++) {
        const j = d[k][1], key = Math.min(i, j) + '-' + Math.max(i, j);
        if (!seen.has(key)) { seen.add(key); edges.push([i, j]); }
      }
    }
    // three great-circle rings at varied tilts
    const rings = [];
    [[0.5, 0.2, 1.0], [1.1, 0.9, 0.98], [2.0, 0.4, 1.02]].forEach(([tx, tz, rad]) => {
      const pts = [];
      for (let k = 0; k < 60; k++) {
        const an = 2 * Math.PI * k / 60;
        let x = Math.cos(an) * rad, y = Math.sin(an) * rad, z = 0;
        let y1 = y * Math.cos(tx) - z * Math.sin(tx), z1 = y * Math.sin(tx) + z * Math.cos(tx);
        let x2 = x * Math.cos(tz) - y1 * Math.sin(tz), y2 = x * Math.sin(tz) + y1 * Math.cos(tz);
        pts.push({ x: x2, y: y2, z: z1 });
      }
      rings.push(pts);
    });
    this.nodes = nodes; this.edges = edges; this.rings = rings;
    this.pulses = [];
    for (let i = 0; i < 18; i++) this._spawn();
  }
  _spawn() { const e = this.edges; if (e && e.length) this.pulses.push({ ei: (Math.random() * e.length) | 0, p: Math.random(), sp: 0.4 + Math.random() * 0.5 }); }

  // ---- energy envelope ----
  // Returns 0..1 "loudness". If a mic/level is supplied it's used directly;
  // otherwise a synthetic per-state envelope animates the orb.
  _energy(t) {
    if (this._analyser) {
      this._analyser.getByteFrequencyData(this._freq);
      let sum = 0; for (let i = 0; i < this._freq.length; i++) sum += this._freq[i];
      const lvl = sum / (this._freq.length * 255);
      this._micLevel = this._micLevel == null ? lvl : this._micLevel * 0.7 + lvl * 0.3; // smooth
      return Math.min(1, this._micLevel * 1.6);
    }
    if (this._micLevel != null) return this._micLevel;
    const a = this.anim;
    if (a === 'speaking') {
      const f = 0.5 + 0.5 * Math.sin(t * 9.1), s = 0.6 + 0.4 * Math.sin(t * 2.7 + 1),
            w = 0.45 + 0.55 * Math.max(0, Math.sin(t * 1.35 + 0.3));
      return Math.min(1, f * s * w * 1.35);
    }
    if (a === 'listening') return 0.32 + 0.12 * Math.sin(t * 4) + 0.06 * Math.sin(t * 7.1);
    if (a === 'thinking')  return 0.2 + 0.06 * Math.sin(t * 3.2);
    return 0.12 + 0.05 * Math.sin(t * 1.4);
  }

  _rot(n, yaw, pitch) {
    let x = n.x * Math.cos(yaw) - n.z * Math.sin(yaw), z1 = n.x * Math.sin(yaw) + n.z * Math.cos(yaw);
    let y = n.y * Math.cos(pitch) - z1 * Math.sin(pitch), z = n.y * Math.sin(pitch) + z1 * Math.cos(pitch);
    return { x, y, z };
  }
  _rgba(hex, a) { const h = hex.replace('#', ''); return `rgba(${parseInt(h.slice(0, 2), 16)},${parseInt(h.slice(2, 4), 16)},${parseInt(h.slice(4, 6), 16)},${a})`; }

  // ---- render loop ----
  _loop(now) {
    const ctx = this.ctx;
    if (ctx && this.nodes) {
      const t = (now - this._t0) / 1000, w = this._w, h = this._h, cx = w / 2, cy = h / 2 - 6;
      const e = this._energy(t), P = this.C.palettes[this.accent];
      const R = Math.min(w, h) * 0.345 * (1 + e * 0.05), fov = 2.7;
      const rot = this.C.rotation;
      const yaw = t * rot.yaw, pitch = rot.pitchBase + Math.sin(t * 0.12) * rot.pitchWobble;
      ctx.clearRect(0, 0, w, h);
      ctx.globalCompositeOperation = 'lighter';   // additive glow

      // project nodes to screen
      const pr = this.nodes.map(n => {
        const u = this._rot(n, yaw, pitch);
        const breath = 1 + e * 0.06 * Math.sin(t * 3 + n.ph);
        const sc = fov / (fov - u.z);
        const d = (u.z + 1) / 2;                    // 0 = far, 1 = near
        return { sx: cx + u.x * R * sc * breath, sy: cy + u.y * R * sc * breath, d, n };
      });

      // rings
      this.rings.forEach((pts, ri) => {
        ctx.beginPath();
        pts.forEach((p, k) => { const u = this._rot(p, yaw + ri * 0.3, pitch); const sc = fov / (fov - u.z); const X = cx + u.x * R * sc, Y = cy + u.y * R * sc; k ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
        ctx.closePath(); ctx.strokeStyle = this._rgba(P.ring, 0.10 + e * 0.12); ctx.lineWidth = 1; ctx.stroke();
      });

      // neural edges (depth-faded)
      for (let k = 0; k < this.edges.length; k++) {
        const a = pr[this.edges[k][0]], b = pr[this.edges[k][1]];
        const da = (a.d + b.d) / 2, al = da * da * (0.16 + e * 0.4);
        if (al < 0.012) continue;
        ctx.beginPath(); ctx.moveTo(a.sx, a.sy); ctx.lineTo(b.sx, b.sy);
        ctx.strokeStyle = this._rgba(P.edge, al); ctx.lineWidth = 0.55 * da + 0.2; ctx.stroke();
      }

      // nodes — back-to-front, with depth-of-field bokeh on the far ones
      const dof = this.C.depthOfField;
      const order = pr.map((p, i) => i).sort((i, j) => pr[i].d - pr[j].d);
      for (let oi = 0; oi < order.length; oi++) {
        const p = pr[order[oi]], sz = (0.5 + p.d * 1.9) * (1 + e * 0.35);
        const blur = dof.enabled ? Math.max(0, (dof.nearPlane - p.d)) / dof.nearPlane * dof.maxBlur : 0;
        const dim = dof.enabled ? 1 - (1 - Math.min(1, p.d / dof.nearPlane)) * dof.farDim : 1;
        if (blur > 0.4) { ctx.shadowColor = this._rgba(P.node, (0.5 + p.d * 0.5) * dim); ctx.shadowBlur = blur; }
        const rad = blur > 0.4 ? sz * (1 + blur * 0.16) : sz;
        ctx.beginPath(); ctx.arc(p.sx, p.sy, rad, 0, 6.2832); ctx.fillStyle = this._rgba(P.node, (0.25 + p.d * 0.75) * dim); ctx.fill();
        ctx.shadowBlur = 0;
        if (p.d > 0.82) { ctx.beginPath(); ctx.arc(p.sx, p.sy, sz * 2.4, 0, 6.2832); ctx.fillStyle = this._rgba(P.node, 0.12 + e * 0.1); ctx.fill(); }
      }

      // signal pulses travelling along edges
      const stC = this.C.states[this.anim] || this.C.states.idle;
      while (this.pulses.length < stC.pulses) this._spawn();
      const bright = stC.brightness, spd = 0.5 + e * 1.4;
      for (let i = this.pulses.length - 1; i >= 0; i--) {
        const pu = this.pulses[i]; pu.p += pu.sp * spd * 0.016;
        if (pu.p >= 1) { this.pulses.splice(i, 1); continue; }
        const a = pr[this.edges[pu.ei][0]], b = pr[this.edges[pu.ei][1]];
        const x = a.sx + (b.sx - a.sx) * pu.p, y = a.sy + (b.sy - a.sy) * pu.p;
        const fade = Math.sin(pu.p * Math.PI), d = (a.d + b.d) / 2;
        ctx.beginPath(); ctx.arc(x, y, 1.6 + fade * 1.4, 0, 6.2832); ctx.fillStyle = this._rgba(P.core, (0.5 + 0.5 * d) * fade * bright); ctx.fill();
        ctx.beginPath(); ctx.arc(x, y, 4.5 * fade, 0, 6.2832); ctx.fillStyle = this._rgba(P.edge, 0.25 * fade * bright); ctx.fill();
      }

      // core glow
      const cr = R * (0.34 + e * 0.4);
      const cg = ctx.createRadialGradient(cx, cy, 1, cx, cy, cr);
      cg.addColorStop(0, this._rgba(P.core, 0.6 + e * 0.3)); cg.addColorStop(0.4, this._rgba(P.edge, 0.28)); cg.addColorStop(1, this._rgba(P.edge, 0));
      ctx.beginPath(); ctx.arc(cx, cy, cr, 0, 6.2832); ctx.fillStyle = cg; ctx.fill();

      // outer HUD arcs
      ctx.lineWidth = 1.2;
      [[1.16, 0.6], [1.28, -0.35]].forEach(ar => { const a0 = t * ar[1]; for (let s = 0; s < 3; s++) { const st = a0 + s * 2.094; ctx.beginPath(); ctx.arc(cx, cy, R * ar[0], st, st + 1.2); ctx.strokeStyle = this._rgba(P.ring, 0.14 + e * 0.1); ctx.stroke(); } });

      ctx.globalCompositeOperation = 'source-over';
    }
    this._raf = requestAnimationFrame(this._loop);
  }
}

function deepMerge(base, over) {
  const out = Array.isArray(base) ? base.slice() : Object.assign({}, base);
  for (const k in over) {
    if (over[k] && typeof over[k] === 'object' && !Array.isArray(over[k]) && typeof out[k] === 'object') out[k] = deepMerge(out[k], over[k]);
    else out[k] = over[k];
  }
  return out;
}
