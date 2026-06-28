/*
 * EVE window controller.
 *
 * Owns the small UI state (mode / muted / anim), renders the orb via EveOrb,
 * drives orb energy from the real microphone, and — when the Python agent is
 * running — lets the agent override `anim` over a Server-Sent Events stream so
 * the orb mirrors the live listening / thinking / speaking pipeline.
 *
 * Without a backend the window is fully self-contained (exactly like the design
 * prototype): every control drives the orb locally with the synthetic envelope.
 */

import { EveOrb } from './eve-orb.js';

// ---- palette table (for retinting the glass shell per accent) ----
const PALETTES = {
  amber:  { edge: '#ff8c1a', ring: '#ffb347', hex: '#ffb65e' },
  cyan:   { edge: '#1fb0e0', ring: '#52d4f0', hex: '#6fdcf5' },
  violet: { edge: '#8a6cff', ring: '#a98cff', hex: '#b49cff' },
  mono:   { edge: '#9fb0c8', ring: '#b8c6da', hex: '#cdd8e6' },
};

const STATUS_TEXT = {
  idle: 'READY',
  listening: 'LISTENING',
  thinking: 'THINKING',
  speaking: 'SPEAKING',
};

// ---- load config (JSONC → JSON) -------------------------------------------
async function loadConfig() {
  try {
    const res = await fetch('./eve.config.json', { cache: 'no-store' });
    if (!res.ok) return {};
    let txt = await res.text();
    txt = txt.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/.*$/gm, '$1');
    return JSON.parse(txt);
  } catch {
    return {}; // renderer falls back to its built-in defaults
  }
}

function rgba(hex, a) {
  const h = hex.replace('#', '');
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}

class EveWindow {
  constructor(config) {
    this.config = config;
    this.canvas = document.getElementById('eve-canvas');
    this.orb = new EveOrb(this.canvas, config);

    // UI state
    this.mode = 'ptt';            // 'ptt' | 'always'
    this.muted = false;
    this.anim = 'idle';           // mirrors orb state
    this.accent = config.defaultAccent || 'amber';
    this.timings = config.timings || { thinkingDelayMs: 650, replyDurationMs: 3200 };
    this._timers = [];
    this._agentDriven = false;    // true once the backend pushes a state

    this._cacheEls();
    this._bind();
    this.setAccent(this.accent);
    this.setMode('ptt');
    this._connectAgent();

    window.addEventListener('resize', () => this.orb.resize());
  }

  _cacheEls() {
    this.el = {
      segPtt: document.getElementById('seg-ptt'),
      segAlways: document.getElementById('seg-always'),
      ptt: document.getElementById('ptt-button'),
      pttLabel: document.getElementById('ptt-label'),
      always: document.getElementById('always-controls'),
      mute: document.getElementById('mute-button'),
      demo: document.getElementById('demo-button'),
      statusDot: document.getElementById('status-dot'),
      statusLabel: document.getElementById('status-label'),
      chips: Array.from(document.querySelectorAll('.eve-chip')),
    };
  }

  _bind() {
    this.el.segPtt.addEventListener('click', () => this.setMode('ptt'));
    this.el.segAlways.addEventListener('click', () => this.setMode('always'));

    // Push-to-talk: hold to listen, release to send.
    const press = (e) => { e.preventDefault(); this._ensureMic(); this.press(); };
    const release = () => this.release();
    this.el.ptt.addEventListener('mousedown', press);
    this.el.ptt.addEventListener('mouseup', release);
    this.el.ptt.addEventListener('mouseleave', release);
    this.el.ptt.addEventListener('touchstart', press, { passive: false });
    this.el.ptt.addEventListener('touchend', release);

    this.el.mute.addEventListener('click', () => this.toggleMute());
    this.el.demo.addEventListener('click', () => { this._ensureMic(); this.demoReply(); });

    this.el.chips.forEach((chip) =>
      chip.addEventListener('click', () => this.setAnim(chip.dataset.state))
    );
  }

  // ---- accent / palette ----------------------------------------------------
  setAccent(key) {
    if (!PALETTES[key]) return;
    this.accent = key;
    this.orb.setAccent(key);
    const p = PALETTES[key];
    const root = document.documentElement.style;
    root.setProperty('--accent-hex', p.hex);
    root.setProperty('--accent-edge', p.edge);
    root.setProperty('--accent-ring', p.ring);
    root.setProperty('--blob-a', rgba(p.edge, 0.16));
    root.setProperty('--blob-b', rgba(p.ring, 0.10));
    root.setProperty('--halo', rgba(p.edge, 0.10));
  }

  // ---- orb state -----------------------------------------------------------
  setAnim(name) {
    this._clearTimers();
    this.anim = name;
    this.orb.setState(name);
    this._render();
  }

  _render() {
    const accentHex = PALETTES[this.accent].hex;
    const dotColor = {
      idle: this.muted ? '#ff6b6b' : 'rgba(238,241,246,0.42)',
      listening: '#3ad29b',
      thinking: '#ffc44d',
      speaking: accentHex,
    }[this.anim];
    const label = this.anim === 'idle' && this.muted ? 'MUTED' : STATUS_TEXT[this.anim];

    this.el.statusDot.style.color = dotColor;
    this.el.statusDot.style.background = dotColor;
    this.el.statusLabel.textContent = label;

    // segments
    this.el.segPtt.setAttribute('aria-selected', String(this.mode === 'ptt'));
    this.el.segAlways.setAttribute('aria-selected', String(this.mode === 'always'));

    // ptt held appearance
    const held = this.mode === 'ptt' && this.anim === 'listening';
    this.el.ptt.classList.toggle('is-held', held);
    this.el.pttLabel.textContent = held ? 'Listening… release to send' : 'Hold to talk';

    // mute appearance
    this.el.mute.classList.toggle('is-muted', this.muted);
    this.el.mute.textContent = this.muted ? 'Unmute' : 'Mute mic';

    // preview chips
    this.el.chips.forEach((chip) =>
      chip.classList.toggle('is-active', chip.dataset.state === this.anim)
    );
  }

  // ---- mode / interactions (mirror the prototype) --------------------------
  setMode(mode) {
    this._clearTimers();
    this.mode = mode;
    this.muted = false;
    this.el.ptt.hidden = mode !== 'ptt';
    this.el.always.hidden = mode !== 'always';
    // PTT rests at idle; Always-on begins listening.
    this.setAnim(mode === 'ptt' ? 'idle' : 'listening');
    if (mode === 'always') this._ensureMic();
  }

  press() {
    this._clearTimers();
    this.setAnim('listening');
  }

  release() {
    if (this.anim !== 'listening' || this.mode !== 'ptt') return;
    this._clearTimers();
    this.setAnim('thinking');
    const { thinkingDelayMs, replyDurationMs } = this.timings;
    this._timers.push(setTimeout(() => this.setAnim('speaking'), thinkingDelayMs));
    this._timers.push(setTimeout(() => this.setAnim('idle'), thinkingDelayMs + replyDurationMs));
  }

  demoReply() {
    this._clearTimers();
    this.muted = false;
    this.setAnim('thinking');
    const { thinkingDelayMs, replyDurationMs } = this.timings;
    this._timers.push(setTimeout(() => this.setAnim('speaking'), thinkingDelayMs - 50));
    this._timers.push(setTimeout(() => this.setAnim('listening'), thinkingDelayMs - 50 + replyDurationMs));
  }

  toggleMute() {
    this._clearTimers();
    this.muted = !this.muted;
    this.setAnim(this.muted ? 'idle' : 'listening');
  }

  _clearTimers() {
    this._timers.forEach(clearTimeout);
    this._timers = [];
  }

  // ---- real microphone → orb energy ----------------------------------------
  async _ensureMic() {
    if (this._micRequested) return;
    this._micRequested = true;
    try {
      const ac = new (window.AudioContext || window.webkitAudioContext)();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const src = ac.createMediaStreamSource(stream);
      const analyser = ac.createAnalyser();
      analyser.fftSize = 256;
      src.connect(analyser);
      this.orb.attachMic(analyser);
    } catch {
      // Mic denied/unavailable → orb keeps using its synthetic envelope.
      this._micRequested = false;
    }
  }

  // ---- agent bridge (Server-Sent Events) -----------------------------------
  // When the Python agent runs it pushes {state, accent} here so the orb mirrors
  // the live pipeline. Falls back silently to local demo control if absent.
  _connectAgent() {
    let es;
    try {
      es = new EventSource('./events');
    } catch {
      return;
    }
    es.addEventListener('state', (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.accent) this.setAccent(data.accent);
        if (data.state && STATUS_TEXT[data.state]) {
          this._agentDriven = true;
          this._clearTimers();
          this.muted = false;
          this.anim = data.state;
          this.orb.setState(data.state);
          this._render();
        }
      } catch {
        /* ignore malformed frames */
      }
    });
    es.onerror = () => { /* EventSource auto-reconnects; nothing to do */ };
  }
}

loadConfig().then((config) => {
  // eslint-disable-next-line no-new
  new EveWindow(config);
});
