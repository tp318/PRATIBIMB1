/*
 * AeroTwin-4 Engine Health Monitoring System, operator interface.
 *
 * React is loaded from a locally vendored UMD build rather than a CDN: a live
 * demo must not depend on venue internet, and there is no build step for the same
 * reason, so one uvicorn command serves the API and this page together.
 *
 * Charts are drawn on plain canvas rather than a charting library, so there is one
 * fewer dependency that can fail to load before a demonstration.
 */

const { useState, useEffect, useRef, useCallback } = React;
const h = React.createElement;

const MAX_POINTS = 300;
const WS_URL = "ws://" + location.host + "/ws/telemetry";

// ----------------------------------------------------------- SortieLogModal
// Columns shown in the sortie live log table.
// Each entry: { key, label, unit, digits, dtKey, rKey, zKey }
const LOG_COLS = [
  { key: "rpm",           label: "RPM",       unit: "rpm",  digits: 0, dtKey: "rpm",           rKey: "rpm",      zKey: "rpm" },
  { key: "cht",           label: "CHT",       unit: "°C",   digits: 1, dtKey: "cht",           rKey: "cht",      zKey: "cht" },
  { key: "egt",           label: "EGT",       unit: "°C",   digits: 1, dtKey: "egt",           rKey: "egt",      zKey: "egt" },
  { key: "oil_pressure",  label: "Oil Press", unit: "bar",  digits: 3, dtKey: "oil_pressure",  rKey: "oil_press",zKey: "oil_press" },
  { key: "oil_temperature",label:"Oil Temp",  unit: "°C",   digits: 1, dtKey: "oil_temperature",rKey: "oil_temp", zKey: "oil_temp" },
  { key: "fuel_flow_lph", label: "Fuel Flow", unit: "L/h",  digits: 2, dtKey: "fuel_flow_lph", rKey: "fuel_flow",zKey: "fuel_flow" },
];

function zClass(z) {
  if (z === null || z === undefined || !Number.isFinite(z)) return "";
  const a = Math.abs(z);
  if (a >= 3) return "z-warn";
  if (a >= 2) return "z-caut";
  return "z-ok";
}

function downloadLogCSV(rows) {
  if (!rows || rows.length === 0) {
    alert("No sortie telemetry rows logged yet. Start a sortie first!");
    return;
  }
  const headers = [
    "timestamp_s", "throttle", "altitude_ft",
    "rpm_real", "rpm_dt", "rpm_z",
    "cht_c_real", "cht_c_dt", "cht_z",
    "egt_c_real", "egt_c_dt", "egt_z",
    "oil_press_bar_real", "oil_press_bar_dt", "oil_press_z",
    "oil_temp_c_real", "oil_temp_c_dt", "oil_temp_z",
    "fuel_flow_lph_real", "fuel_flow_lph_dt", "fuel_flow_z"
  ];
  const csvLines = [headers.join(",")];
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    const real = r.real || {};
    const dt = r.dt || {};
    const z = r.z || {};
    const line = [
      r.t != null ? Number(r.t).toFixed(2) : "",
      r.thr != null ? Number(r.thr).toFixed(3) : "",
      r.alt != null ? Number(r.alt).toFixed(1) : "0",
      real.rpm != null ? Number(real.rpm).toFixed(1) : "",
      dt.rpm != null ? Number(dt.rpm).toFixed(1) : "",
      z.rpm != null ? Number(z.rpm).toFixed(3) : "",
      real.cht != null ? Number(real.cht).toFixed(2) : "",
      dt.cht != null ? Number(dt.cht).toFixed(2) : "",
      z.cht != null ? Number(z.cht).toFixed(3) : "",
      real.egt != null ? Number(real.egt).toFixed(2) : "",
      dt.egt != null ? Number(dt.egt).toFixed(2) : "",
      z.egt != null ? Number(z.egt).toFixed(3) : "",
      real.oil_pressure != null ? Number(real.oil_pressure).toFixed(3) : "",
      dt.oil_pressure != null ? Number(dt.oil_pressure).toFixed(3) : "",
      z.oil_press != null ? Number(z.oil_press).toFixed(3) : "",
      real.oil_temperature != null ? Number(real.oil_temperature).toFixed(2) : "",
      dt.oil_temperature != null ? Number(dt.oil_temperature).toFixed(2) : "",
      z.oil_temp != null ? Number(z.oil_temp).toFixed(3) : "",
      real.fuel_flow_lph != null ? Number(real.fuel_flow_lph).toFixed(2) : "",
      dt.fuel_flow_lph != null ? Number(dt.fuel_flow_lph).toFixed(2) : "",
      z.fuel_flow != null ? Number(z.fuel_flow).toFixed(3) : "",
    ];
    csvLines.push(line.join(","));
  }
  const blob = new Blob([csvLines.join("\r\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "sortie_telemetry_log_" + Math.floor(Date.now() / 1000) + ".csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function SortieLogModal(props) {
  const { open, onClose, rows } = props;
  const tbodyRef = useRef(null);
  const [autoscroll, setAutoscroll] = useState(true);

  useEffect(function () {
    if (autoscroll && tbodyRef.current) {
      tbodyRef.current.scrollTop = tbodyRef.current.scrollHeight;
    }
  }, [rows.length, autoscroll]);

  if (!open) return null;

  return h("div", { className: "modal-overlay", onClick: function(e){ if(e.target===e.currentTarget) onClose(); } },
    h("div", { className: "modal-box" },
      h("div", { className: "modal-head" },
        h("span", { className: "modal-title" }, "Sortie Live Log — Real Engine vs Digital Twin"),
        h("span", { className: "modal-meta" }, rows.length + " frames @ 10 Hz"),
        h("div", { className: "modal-actions" },
          h("label", { className: "modal-toggle" },
            h("input", { type: "checkbox", checked: autoscroll, onChange: function(){ setAutoscroll(function(v){ return !v; }); } }),
            " Auto-scroll"
          ),
          h("button", {
            className: "btn-dl",
            onClick: function() { downloadLogCSV(rows); },
            title: "Download recorded sortie logs as CSV",
          }, "⬇ Download CSV"),
          h("button", { className: "modal-close", onClick: onClose }, "✕ Close")
        )
      ),
      h("div", { className: "modal-table-wrap", ref: tbodyRef },
        h("table", { className: "log-table" },
          h("thead", null,
            h("tr", null,
              h("th", null, "Time"),
              h("th", null, "Throttle"),
              h("th", null, "Alt ft"),
              LOG_COLS.map(function(col) {
                return [
                  h("th", { key: col.key+"_r" }, "Real " + col.label),
                  h("th", { key: col.key+"_d" }, "DT " + col.label),
                  h("th", { key: col.key+"_z", className: "th-z" }, "z"),
                ];
              })
            )
          ),
          h("tbody", null,
            rows.slice(-400).map(function(row, i) {
              return h("tr", { key: i },
                h("td", { className: "mono" }, fmt(row.t, 1)),
                h("td", { className: "mono" }, fmt(row.thr, 2)),
                h("td", { className: "mono" }, fmt(row.alt, 0)),
                LOG_COLS.map(function(col) {
                  const rv = row.real && row.real[col.key];
                  const dv = row.dt   && row.dt[col.dtKey];
                  const zv = row.z    && row.z[col.zKey];
                  return [
                    h("td", { key: col.key+"_r", className: "num" }, fmt(rv, col.digits)),
                    h("td", { key: col.key+"_d", className: "num" }, fmt(dv, col.digits)),
                    h("td", { key: col.key+"_z", className: "num " + zClass(zv) }, fmt(zv, 2)),
                  ];
                })
              );
            })
          )
        )
      )
    )
  );
}

// ----------------------------------------------------------- Avionics Urgent Warning Buzzer Synthesizer
let _audioCtx = null;
function playUrgentBuzzerBurst() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    if (!_audioCtx) {
      _audioCtx = new AudioCtx();
    }
    if (_audioCtx.state === "suspended") {
      _audioCtx.resume();
    }
    const now = _audioCtx.currentTime;

    // Avionics Master Warning Buzzer:
    // Triple rapid dissonant burst: 780 Hz & 980 Hz sawtooth waveforms (dissonant harmonic buzz)
    const bursts = [0.0, 0.11, 0.22];
    bursts.forEach(function(offset) {
      const tStart = now + offset;
      const tEnd = tStart + 0.085;

      // Tone 1: 780 Hz sawtooth
      const osc1 = _audioCtx.createOscillator();
      const gain1 = _audioCtx.createGain();
      osc1.type = "sawtooth";
      osc1.frequency.setValueAtTime(780, tStart);
      gain1.gain.setValueAtTime(0.001, tStart);
      gain1.gain.linearRampToValueAtTime(0.22, tStart + 0.008);
      gain1.gain.exponentialRampToValueAtTime(0.001, tEnd);
      osc1.connect(gain1);
      gain1.connect(_audioCtx.destination);
      osc1.start(tStart);
      osc1.stop(tEnd);

      // Tone 2: 980 Hz sawtooth
      const osc2 = _audioCtx.createOscillator();
      const gain2 = _audioCtx.createGain();
      osc2.type = "sawtooth";
      osc2.frequency.setValueAtTime(980, tStart);
      gain2.gain.setValueAtTime(0.001, tStart);
      gain2.gain.linearRampToValueAtTime(0.20, tStart + 0.008);
      gain2.gain.exponentialRampToValueAtTime(0.001, tEnd);
      osc2.connect(gain2);
      gain2.connect(_audioCtx.destination);
      osc2.start(tStart);
      osc2.stop(tEnd);
    });
  } catch (e) {
    console.warn("AudioContext error:", e);
  }
}

// ----------------------------------------------------------- Apple Blur Hero Overlay (Minimalist Typography)
const HERO_TEXT = "PROJECT PRATIBIMB: A Physics Informed Digital Twin For Health Monitoring and Predictive Maintenance of MALE UAVs";

function HeroOverlay(props) {
  const { onStartDemo, onSkip } = props;
  const [typedIndex, setTypedIndex] = useState(0);
  const [unblurring, setUnblurring] = useState(false);

  useEffect(function() {
    if (typedIndex < HERO_TEXT.length) {
      const timer = setTimeout(function() {
        setTypedIndex(function(prev) { return prev + 1; });
      }, 22);
      return function() { clearTimeout(timer); };
    }
  }, [typedIndex]);

  function handleStart() {
    setUnblurring(true);
    setTimeout(function() {
      onStartDemo();
    }, 550);
  }

  function handleSkip() {
    setUnblurring(true);
    setTimeout(function() {
      onSkip();
    }, 380);
  }

  return h("div", { className: cls("pratibimb-hero-overlay", unblurring && "unblurring") },
    h("div", { className: "hero-minimal-wrap" },
      h("h1", { className: "hero-apple-title" }, "PRATIBIMB"),
      h("p", { className: "hero-typewriter-line" },
        HERO_TEXT.slice(0, typedIndex),
        typedIndex < HERO_TEXT.length ? h("span", { className: "typewriter-cursor" }) : null
      ),
      h("div", { className: "hero-actions" },
        h("button", {
          className: "hero-minimal-btn",
          onClick: handleStart,
          id: "hero-start-demo-btn",
        }, "Start Interactive Demo →"),
        h("button", {
          className: "hero-minimal-skip",
          onClick: handleSkip,
          id: "hero-skip-btn",
        }, "Skip to Dashboard")
      )
    )
  );
}

// ----------------------------------------------------------- Interactive Skippable Tutorial
const TUTORIAL_STEPS = [
  {
    step: 1,
    title: "Avionics & Subsystem Link Indicators",
    tag: "Hardware Connectivity",
    desc: "Real-time telemetry heartbeat status for ECU, FADEC, Edge AI compute, Telemetry RF downlink, and Ground Control Station (GCS). Displays dual-channel bus latency in milliseconds and packet counters.",
  },
  {
    step: 2,
    title: "Mission Sortie & Dynamic Flight Controls",
    tag: "Simulation Controls",
    desc: "Launch the engine sortie in Auto-Mission Profile or take manual authority using the interactive throttle, altitude, and ambient temperature sliders.",
  },
  {
    step: 3,
    title: "Virtual Digital Twin & Governing Equations",
    tag: "Physics Engine",
    desc: "Switch to the new 'Digital Twin & Physics' tab to view an interactive virtual schematic of the 4-cylinder engine with live reciprocating pistons, animated airflow, and mathematical differential equations evaluated in real time.",
  },
  {
    step: 4,
    title: "Sensor Telemetry vs Theoretical Residuals",
    tag: "Analytical Redundancy",
    desc: "The digital twin reference model predicts nominal engine behavior. Residual differences (r = Y_measured - Y_twin) decouple true mechanical/thermal faults from ambient atmospheric variations.",
  },
  {
    step: 5,
    title: "Explainable AI (TreeSHAP & DeepSHAP)",
    tag: "Model Transparency",
    desc: "Explore the ML Diagnostics tab powered by TreeSHAP. Observe game-theoretic Shapley feature attributions revealing exactly which physical sensors and residual trends pushed the model toward the diagnosed fault.",
  },
  {
    step: 6,
    title: "Real-Time Fault Injection & Sound Alert",
    tag: "Failure Mode Testing",
    desc: "Use the Fault Injection console to inject Cooling, Bearing, Lubrication, or Cylinder degradation. Watch the dual-tone audio beep beep sound alert activate and the twin track RUL degradation.",
  },
];

function TutorialOverlay(props) {
  const { step, onNext, onPrev, onSkip, onClose } = props;
  const curr = TUTORIAL_STEPS[step] || TUTORIAL_STEPS[0];
  const isLast = step >= TUTORIAL_STEPS.length - 1;

  return h("div", { className: "tutorial-overlay", onClick: function(e) { if(e.target===e.currentTarget) onClose(); } },
    h("div", { className: "tutorial-card" },
      h("div", { className: "tutorial-step-tag" }, curr.tag + " • Step " + (step + 1) + " of " + TUTORIAL_STEPS.length),
      h("h2", { className: "tutorial-step-title" }, curr.title),
      h("p", { className: "tutorial-step-desc" }, curr.desc),
      h("div", { className: "tutorial-footer" },
        h("div", { className: "tutorial-dots" },
          TUTORIAL_STEPS.map(function(_, idx) {
            return h("div", { key: idx, className: cls("tutorial-dot", idx === step && "active") });
          })
        ),
        h("div", { className: "tutorial-nav" },
          h("button", { onClick: onSkip, className: "hero-skip-btn" }, "Skip Tutorial"),
          step > 0 ? h("button", { onClick: onPrev }, "← Back") : null,
          h("button", { className: "go", onClick: isLast ? onClose : onNext }, isLast ? "✓ Get Started" : "Next →")
        )
      )
    )
  );
}

// ----------------------------------------------------------- Subsystem Status Strip
function SubsystemStrip(props) {
  const { subsystems, running, audioEnabled, onToggleAudio, isBuzzerActive } = props;
  const subs = subsystems || {
    ECU: { connected: running, ping_ms: 2.3, status: running ? "ONLINE" : "STANDBY" },
    FADEC: { connected: running, ping_ms: 4.1, status: running ? "ONLINE" : "STANDBY" },
    EDGE: { connected: running, ping_ms: 11.5, status: running ? "ONLINE" : "STANDBY" },
    TELEMETRY: { connected: running, ping_ms: 41.8, status: running ? "ONLINE" : "STANDBY" },
    GCS: { connected: running, ping_ms: 18.2, status: running ? "ONLINE" : "STANDBY" },
  };

  const keys = ["ECU", "FADEC", "EDGE", "TELEMETRY", "GCS"];

  return h("div", { className: "subsystem-strip" },
    h("span", { className: "subsystem-title" }, "Subsystems:"),
    keys.map(function(key) {
      const item = subs[key] || {};
      const isOnline = item.connected || (running && item.status === "ONLINE");
      return h("div", { key: key, className: cls("subsystem-pill", isOnline ? "online" : "standby"), title: item.detail || key },
        h("span", { className: "dot" }),
        h("span", null, key),
        isOnline && item.ping_ms ? h("span", { className: "subsystem-ms" }, item.ping_ms + "ms") : h("span", { className: "subsystem-ms" }, "STANDBY")
      );
    }),
    h("button", {
      className: cls("audio-toggle-btn", audioEnabled && "active", isBuzzerActive && "buzzing"),
      onClick: onToggleAudio,
      title: isBuzzerActive ? "Mute active continuous avionics buzzer" : "Toggle avionics fault warning buzzer",
    },
      isBuzzerActive ? "🚨 Alarm Active (Mute)" : (audioEnabled ? "🔊 Alarm: ON" : "🔇 Alarm: MUTED")
    )
  );
}

const INK = "#1b2027";
const ACCENT = "#1f5fa8";
const GRID = "#e4e7ec";

// ------------------------------------------------------------------ helpers

function cls() {
  return Array.prototype.slice.call(arguments).filter(Boolean).join(" ");
}

function fmt(v, d, fallback) {
  if (d === undefined) d = 1;
  if (v === null || v === undefined || Number.isNaN(v)) return fallback === undefined ? "–" : fallback;
  return Number(v).toFixed(d);
}

function pct(v, d) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  return Number(v).toFixed(d === undefined ? 1 : d) + "%";
}

function clockFrom(seconds) {
  if (seconds === null || seconds === undefined) return "00:00.0";
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return String(m).padStart(2, "0") + ":" + s.toFixed(1).padStart(4, "0");
}

function healthState(v) {
  if (v === null || v === undefined) return "idle";
  if (v >= 0.7) return "normal";
  if (v >= 0.35) return "caution";
  return "warning";
}

const REC_TEXT = {
  GO: "Cleared for dispatch",
  GO_WITH_MONITORING: "Dispatch with monitoring",
  ABORT_OR_SHORTEN: "Shorten or abort sortie",
  NO_GO: "Not cleared for dispatch",
};

const REC_STATE = {
  GO: "normal",
  GO_WITH_MONITORING: "caution",
  ABORT_OR_SHORTEN: "warning",
  NO_GO: "warning",
};

const SEV_STATE = { INFO: "info", CAUTION: "caution", WARNING: "warning" };
const PRIO_STATE = {
  ROUTINE: "info",
  SCHEDULED: "accent",
  PRIORITY: "caution",
  IMMEDIATE: "warning",
};

// ------------------------------------------------------------------- charts

function Trace(props) {
  const ref = useRef(null);
  const { observed, expected, tick, wide } = props;

  // `tick` belongs in the dependency list. History arrays are mutated in place for
  // speed, so their identity never changes; depending on the array alone would run
  // this effect once and leave an empty canvas for the rest of the session.
  useEffect(function () {
    const cv = ref.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth;
    const hh = cv.clientHeight;
    cv.width = w * dpr;
    cv.height = hh * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, hh);

    const series = [observed, expected].filter(function (s) { return s && s.length > 1; });
    if (!series.length) return;

    const all = [];
    series.forEach(function (s) {
      s.forEach(function (v) { if (Number.isFinite(v)) all.push(v); });
    });
    if (!all.length) return;

    let lo = Math.min.apply(null, all);
    let hi = Math.max.apply(null, all);
    if (hi - lo < 1e-6) { hi += 1; lo -= 1; }
    const pad = (hi - lo) * 0.14;
    lo -= pad; hi += pad;

    ctx.strokeStyle = GRID;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = Math.round((hh / 4) * i) + 0.5;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Scale x to the data actually held, not to buffer capacity, so the trace fills
    // the panel from the first seconds instead of sitting in the left third.
    let span = 1;
    series.forEach(function (s) { span = Math.max(span, s.length - 1); });

    function draw(data, stroke, dashed) {
      if (!data || data.length < 2) return;
      ctx.beginPath();
      ctx.strokeStyle = stroke;
      ctx.lineWidth = dashed ? 1.25 : 1.6;
      ctx.setLineDash(dashed ? [3, 3] : []);
      for (let i = 0; i < data.length; i++) {
        const x = (i / span) * w;
        const y = hh - ((data[i] - lo) / (hi - lo)) * hh;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    draw(expected, ACCENT, true);
    draw(observed, INK, false);
  }, [observed, expected, tick]);

  const last = observed && observed.length ? observed[observed.length - 1] : null;
  const lastExp = expected && expected.length ? expected[expected.length - 1] : null;

  return h("div", { className: cls("chart", wide && "wide") },
    h("div", { className: "chart-top" },
      h("span", { className: "chart-name" }, props.label),
      h("span", { className: "chart-read" },
        fmt(last, props.digits),
        h("span", { className: "u" }, " " + props.unit),
        lastExp === null ? null : h("span", { className: "tw" }, "twin " + fmt(lastExp, props.digits))
      )
    ),
    h("canvas", { ref: ref })
  );
}

/** Single-series chart with a zero reference line, for deviation quantities. */
function DeviationTrace(props) {
  const ref = useRef(null);
  const { data, tick, color } = props;

  useEffect(function () {
    const cv = ref.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, hh = cv.clientHeight;
    cv.width = w * dpr; cv.height = hh * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, hh);

    const vals = (data || []).filter(function (v) { return Number.isFinite(v); });
    if (vals.length < 2) return;

    let lo = Math.min.apply(null, vals);
    let hi = Math.max.apply(null, vals);
    // Always keep zero in frame: a deviation chart that crops the baseline makes
    // a small wobble look like a large excursion.
    lo = Math.min(lo, 0); hi = Math.max(hi, 0);
    if (hi - lo < 1e-6) { hi += 1; lo -= 1; }
    const pad = (hi - lo) * 0.14;
    lo -= pad; hi += pad;

    const yOf = function (v) { return hh - ((v - lo) / (hi - lo)) * hh; };

    ctx.strokeStyle = GRID; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = Math.round((hh / 4) * i) + 0.5;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    const zy = Math.round(yOf(0)) + 0.5;
    ctx.strokeStyle = "#9aa2ae"; ctx.lineWidth = 1; ctx.setLineDash([2, 2]);
    ctx.beginPath(); ctx.moveTo(0, zy); ctx.lineTo(w, zy); ctx.stroke();
    ctx.setLineDash([]);

    const span = Math.max(1, data.length - 1);
    ctx.beginPath();
    ctx.strokeStyle = color || INK;
    ctx.lineWidth = 1.6;
    let started = false;
    for (let i = 0; i < data.length; i++) {
      if (!Number.isFinite(data[i])) continue;
      const x = (i / span) * w, y = yOf(data[i]);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }, [data, tick, color]);

  const last = (function () {
    if (!data) return null;
    for (let i = data.length - 1; i >= 0; i--) if (Number.isFinite(data[i])) return data[i];
    return null;
  })();

  return h("div", { className: cls("chart", props.wide && "wide") },
    h("div", { className: "chart-top" },
      h("span", { className: "chart-name" }, props.label),
      h("span", { className: "chart-read" }, fmt(last, props.digits), h("span", { className: "u" }, " " + props.unit))
    ),
    h("canvas", { ref: ref })
  );
}

// -------------------------------------------------------------- status strip

function StatusStrip(props) {
  const a = props.assessment;
  const health = a && a.health ? a.health.health_index : null;
  const hs = healthState(health);
  const anomaly = (a && a.anomaly) || null;
  const diag = (a && a.diagnosis) || null;
  const rul = (a && a.rul) || null;
  const risk = (a && a.mission_risk) || null;

  const anomState = anomaly ? (anomaly.flagged ? "warning" : "normal") : "idle";
  const anomRatio = anomaly && anomaly.threshold ? Math.min(1, anomaly.score / (anomaly.threshold * 2)) : 0;
  const thrMark = 0.5;

  const diagState = diag ? (diag.predicted_fault === "HEALTHY" || diag.predicted_fault === "NORMAL" ? "normal" : "warning") : "idle";
  const recState = risk ? REC_STATE[risk.recommendation] || "idle" : "idle";

  let rulText = "–", rulNote = "Awaiting first window";
  if (rul) {
    if (rul.error) { rulText = "Error"; rulNote = rul.error; }
    else if (!rul.is_decaying || rul.rul_seconds === null) {
      rulText = "No trend"; rulNote = "No measurable degradation trend";
    } else {
      rulText = fmt(rul.rul_seconds, 0) + " s";
      rulNote = "Band " + fmt(rul.rul_lower_seconds, 0) + " to " +
        (rul.rul_upper_seconds === null ? "unbounded" : fmt(rul.rul_upper_seconds, 0) + " s");
    }
  }

  return h("div", { className: "strip" },
    h("div", { className: "cell" },
      h("div", { className: "cell-label" }, "Engine health index"),
      h("div", { className: cls("cell-value", "v-" + hs) }, health === null ? "–" : (health * 100).toFixed(1) + "%"),
      h("div", { className: "meter" }, h("span", { className: hs, style: { width: (health === null ? 0 : health * 100) + "%" } })),
      h("div", { className: "cell-note" },
        health === null ? "Awaiting first window" :
          hs === "normal" ? "Within normal limits" : hs === "caution" ? "Degraded, above dispatch floor" : "Below dispatch floor")
    ),
    h("div", { className: "cell" },
      h("div", { className: "cell-label" }, "Anomaly detector"),
      h("div", { className: cls("cell-value", "sm", "v-" + anomState) },
        anomaly ? (anomaly.flagged ? "Asserted" : "Nominal") : "–"),
      h("div", { className: "meter" },
        h("span", { className: anomState, style: { width: anomRatio * 100 + "%" } }),
        h("span", { className: "mark", style: { left: thrMark * 100 + "%" } })
      ),
      h("div", { className: "cell-note" },
        anomaly ? "Score " + fmt(anomaly.score, 2) + " of " + fmt(anomaly.threshold, 2) + " limit" : "Awaiting first window")
    ),
    h("div", { className: "cell" },
      h("div", { className: "cell-label" }, "Fault attribution (XGBoost 9-Class)"),
      h("div", { className: cls("cell-value", "sm", "v-" + diagState) },
        diag ? (diag.predicted_fault === "HEALTHY" || diag.predicted_fault === "NORMAL" ? "Normal (No fault)" : diag.predicted_fault.replace(/_/g, " ")) : "–"),
      h("div", { className: "meter" }, h("span", { style: { width: (diag ? diag.confidence * 100 : 0) + "%" } })),
      h("div", { className: "cell-note" },
        diag ? "Confidence " + (diag.confidence * 100).toFixed(0) + "%, runner-up " + (diag.runner_up || "None").toLowerCase().replace(/_/g, " ") : "Awaiting first window")
    ),
    h("div", { className: "cell" },
      h("div", { className: "cell-label" }, "Remaining useful life"),
      h("div", { className: cls("cell-value", "sm", rul && rul.is_decaying ? "v-caution" : "v-normal") }, rulText),
      h("div", { className: "cell-note" }, rulNote)
    ),
    h("div", { className: "cell" },
      h("div", { className: "cell-label" }, "Mission disposition"),
      h("div", { className: cls("cell-value", "sm", "v-" + recState) },
        risk ? REC_TEXT[risk.recommendation] || risk.recommendation : "–"),
      h("div", { className: "cell-note" },
        risk ? "Risk " + risk.risk_band.toLowerCase() + ", score " + fmt(risk.risk_score, 3) : "Awaiting first window")
    )
  );
}

// ------------------------------------------------------------------- tabs

function MonitoringTab(props) {
  const H = props.hist, t = props.telemetry || {}, tick = props.tick;
  const a = props.assessment;
  const risk = (a && a.mission_risk) || null;

  const cyl = [t.cylinder_1_torque, t.cylinder_2_torque, t.cylinder_3_torque, t.cylinder_4_torque];
  let cmax = 1;
  cyl.forEach(function (v) { if (Number.isFinite(v)) cmax = Math.max(cmax, Math.abs(v)); });

  return h("div", { className: "grid2" },
    h("div", null,
      h("div", { className: "block" },
        h("div", { className: "block-head" },
          "Observed against digital twin expected",
          h("span", { className: "legend" },
            h("span", null, h("i", null), "Observed"),
            h("span", null, h("i", { className: "tw" }), "Twin expected")
          )
        ),
        h("div", { className: "charts" },
          h(Trace, { label: "Engine speed", unit: "rpm", digits: 1, observed: H.rpm, expected: H.rpm_exp, tick: tick, wide: true }),
          h(Trace, { label: "Cylinder head temperature", unit: "°C", digits: 1, observed: H.cht, expected: H.cht_exp, tick: tick }),
          h(Trace, { label: "Exhaust gas temperature", unit: "°C", digits: 1, observed: H.egt, expected: H.egt_exp, tick: tick }),
          h(Trace, { label: "Oil pressure", unit: "psi", digits: 1, observed: H.oil, expected: H.oil_exp, tick: tick }),
          h(Trace, { label: "Vibration", unit: "g", digits: 3, observed: H.vib, expected: H.vib_exp, tick: tick })
        )
      )
    ),
    h("div", null,
      h("div", { className: "block" },
        h("div", { className: "block-head" }, "Dispatch assessment"),
        h("div", { className: "block-body" },
          !risk ? h("div", { className: "empty" }, "Awaiting the first assessment window.") :
            h("div", null,
              h("dl", { className: "kv" },
                h("dt", null, "Risk score"), h("dd", null, fmt(risk.risk_score, 3)),
                h("dt", null, "Risk band"), h("dd", null, risk.risk_band),
                h("dt", null, "Health contribution"), h("dd", null, fmt(risk.contributions.health, 3)),
                h("dt", null, "Life contribution"), h("dd", null, fmt(risk.contributions.rul, 3)),
                h("dt", null, "Fault contribution"), h("dd", null, fmt(risk.contributions.fault, 3)),
                h("dt", null, "Anomaly contribution"), h("dd", null, fmt(risk.contributions.anomaly, 3))
              ),
              h("ul", { className: "reasons", style: { marginTop: "12px" } },
                (risk.reasons || []).map(function (r, i) { return h("li", { key: i }, r); }))
            )
        )
      ),
      h("div", { className: "block" },
        h("div", { className: "block-head" }, "Per cylinder torque"),
        h("div", { className: "block-body" },
          h("div", { className: "bars" },
            cyl.map(function (v, i) {
              return h("div", { className: "bar-row", key: i },
                h("span", { className: "nm" }, "C" + (i + 1)),
                h("span", { className: "bar-track" },
                  h("span", { className: v < 0 ? "neg" : "", style: { width: (Math.abs(v || 0) / cmax) * 100 + "%" } })),
                h("span", { className: "vl" }, fmt(v, 1))
              );
            })
          ),
          h("div", { className: "hint" },
            "Instantaneous values. Asymmetry within a single frame reflects the 1-3-4-2 firing order and is not itself a fault indication.")
        )
      ),
      h("div", { className: "block" },
        h("div", { className: "block-head" }, "Engine parameters"),
        h("div", { className: "block-body" },
          h("dl", { className: "kv" },
            h("dt", null, "Operating mode"), h("dd", null, t.operating_mode || "–"),
            h("dt", null, "Throttle"), h("dd", null, fmt(t.throttle, 2)),
            h("dt", null, "Mean torque"), h("dd", null, fmt(t.mean_torque, 1) + " Nm"),
            h("dt", null, "Load torque"), h("dd", null, fmt(t.load_torque, 1) + " Nm"),
            h("dt", null, "Friction torque"), h("dd", null, fmt(t.friction_torque, 2) + " Nm"),
            h("dt", null, "Oil temperature"), h("dd", null, fmt(t.oil_temperature, 1) + " °C"),
            h("dt", null, "Fuel flow"), h("dd", null, fmt(t.fuel_flow_lph, 2) + " L/h"),
            h("dt", null, "Fuel pressure"), h("dd", null, fmt(t.fuel_pressure, 1) + " kPa")
          )
        )
      )
    )
  );
}

function EfficiencyTab(props) {
  const H = props.hist, tick = props.tick, eff = props.efficiency, summary = props.effSummary;

  return h("div", null,
    h("div", { className: "stat-row" },
      h("div", { className: "stat" },
        h("div", { className: "stat-l" }, "Shaft power"),
        h("div", { className: "stat-v" }, fmt(eff && eff.power_kw, 2) + " kW"),
        h("div", { className: "stat-n" }, "Twin expects " + fmt(eff && eff.power_expected_kw, 2) + " kW")),
      h("div", { className: "stat" },
        h("div", { className: "stat-l" }, "Power deficit"),
        h("div", { className: cls("stat-v", eff && eff.power_deficit_pct > 5 ? "v-caution" : null) },
          pct(eff && eff.power_deficit_pct, 2)),
        h("div", { className: "stat-n" }, "Against condition-matched twin")),
      h("div", { className: "stat" },
        h("div", { className: "stat-l" }, "Specific fuel consumption"),
        h("div", { className: "stat-v" }, fmt(eff && eff.bsfc, 3)),
        h("div", { className: "stat-n" }, "kg per kW hour")),
      h("div", { className: "stat" },
        h("div", { className: "stat-l" }, "Fuel penalty"),
        h("div", { className: cls("stat-v", eff && eff.bsfc_penalty_pct > 5 ? "v-caution" : null) },
          pct(eff && eff.bsfc_penalty_pct, 2)),
        h("div", { className: "stat-n" }, "Excess fuel for the same work"))
    ),
    h("div", { className: "block", style: { marginTop: "12px" } },
      h("div", { className: "block-head" },
        "Efficiency trend",
        h("span", { className: "legend" },
          h("span", null, h("i", null), "Observed"),
          h("span", null, h("i", { className: "tw" }), "Twin expected")
        )
      ),
      h("div", { className: "charts" },
        h(Trace, { label: "Shaft power", unit: "kW", digits: 2, observed: H.pw, expected: H.pw_exp, tick: tick, wide: true }),
        h(DeviationTrace, { label: "Power deficit against twin", unit: "%", digits: 2, data: H.pdef, tick: tick, color: "#9a6508" }),
        h(DeviationTrace, { label: "Fuel consumption penalty", unit: "%", digits: 2, data: H.bpen, tick: tick, color: "#a32c22" }),
        h(Trace, { label: "Specific fuel consumption", unit: "kg/kWh", digits: 3, observed: H.bsfc, expected: H.bsfc_exp, tick: tick }),
        h(Trace, { label: "Fuel flow", unit: "L/h", digits: 2, observed: H.fuel, expected: null, tick: tick })
      )
    ),
    h("div", { className: "block" },
      h("div", { className: "block-head" }, "Sortie efficiency summary"),
      h("div", { className: "block-body" },
        !summary || !summary.n_samples ? h("div", { className: "empty" }, "No efficiency samples recorded yet.") :
          h("table", null,
            h("thead", null, h("tr", null,
              h("th", null, "Metric"), h("th", { className: "num" }, "Mean"), h("th", { className: "num" }, "Peak"))),
            h("tbody", null,
              h("tr", null, h("td", null, "Shaft power"), h("td", { className: "num" }, fmt(summary.mean_power_kw, 3) + " kW"), h("td", { className: "num" }, "–")),
              h("tr", null, h("td", null, "Power deficit against twin"), h("td", { className: "num" }, pct(summary.mean_power_deficit_pct, 2)), h("td", { className: "num" }, pct(summary.peak_power_deficit_pct, 2))),
              h("tr", null, h("td", null, "Specific fuel consumption"), h("td", { className: "num" }, fmt(summary.mean_bsfc, 4)), h("td", { className: "num" }, "–")),
              h("tr", null, h("td", null, "Fuel consumption penalty"), h("td", { className: "num" }, pct(summary.mean_bsfc_penalty_pct, 2)), h("td", { className: "num" }, pct(summary.peak_bsfc_penalty_pct, 2)))
            )
          ),
        h("div", { className: "hint" },
          "Efficiency is reported as deviation from the condition-matched twin rather than against a fixed book figure. " +
          "An absolute consumption number cannot be interpreted without knowing ambient conditions and build tolerance; " +
          "the deviation isolates what degradation is costing.")
      )
    )
  );
}

function AlertsTab(props) {
  const alerts = props.alerts || [];
  const counts = props.counts || {};
  return h("div", null,
    h("div", { className: "stat-row" },
      h("div", { className: "stat" },
        h("div", { className: "stat-l" }, "Total events"), h("div", { className: "stat-v" }, alerts.length)),
      h("div", { className: "stat" },
        h("div", { className: "stat-l" }, "Warning"), h("div", { className: "stat-v v-warning" }, counts.WARNING || 0)),
      h("div", { className: "stat" },
        h("div", { className: "stat-l" }, "Caution"), h("div", { className: "stat-v v-caution" }, counts.CAUTION || 0)),
      h("div", { className: "stat" },
        h("div", { className: "stat-l" }, "Information"), h("div", { className: "stat-v" }, counts.INFO || 0))
    ),
    h("div", { className: "block", style: { marginTop: "12px" } },
      h("div", { className: "block-head" }, "Fault and state change log",
        h("span", { className: "aux" }, "Most recent first")),
      !alerts.length ? h("div", { className: "block-body" },
        h("div", { className: "empty" }, "No events recorded. Events are logged when a condition changes, not on every window.")) :
        h("table", null,
          h("thead", null, h("tr", null,
            h("th", { className: "num" }, "Seq"),
            h("th", { className: "num" }, "Sortie time"),
            h("th", null, "Severity"),
            h("th", null, "Category"),
            h("th", null, "Event"),
            h("th", null, "Detail"))),
          h("tbody", null, alerts.map(function (a) {
            return h("tr", { key: a.seq },
              h("td", { className: "num" }, a.seq),
              h("td", { className: "num" }, clockFrom(a.simulation_time)),
              h("td", null, h("span", { className: cls("tag", SEV_STATE[a.severity]) }, a.severity)),
              h("td", null, a.category),
              h("td", null, a.message),
              h("td", { style: { color: "#767f8d" } }, a.detail)
            );
          }))
        )
    )
  );
}

function MaintenanceTab(props) {
  const items = props.items || [];
  const a = props.assessment;
  const diag = (a && a.diagnosis) || null;
  const health = a && a.health ? a.health.health_index : null;

  return h("div", null,
    h("div", { className: "block" },
      h("div", { className: "block-head" }, "Advisory basis"),
      h("div", { className: "block-body" },
        h("dl", { className: "kv", style: { maxWidth: "520px" } },
          h("dt", null, "Attributed subsystem"), h("dd", null, diag ? diag.predicted_fault : "–"),
          h("dt", null, "Attribution confidence"), h("dd", null, diag ? (diag.confidence * 100).toFixed(0) + "%" : "–"),
          h("dt", null, "Runner-up attribution"), h("dd", null, diag ? diag.runner_up : "–"),
          h("dt", null, "Separation margin"), h("dd", null, diag ? fmt(diag.margin, 3) : "–"),
          h("dt", null, "Health index"), h("dd", null, health === null ? "–" : (health * 100).toFixed(1) + "%")
        ),
        diag && diag.margin !== undefined && diag.margin < 0.25 ?
          h("div", { className: "hint", style: { color: "#9a6508" } },
            "Separation between the two leading attributions is narrow. Confirm the indication before committing to component work.") : null
      )
    ),
    h("div", { className: "block" },
      h("div", { className: "block-head" }, "Recommended maintenance actions",
        h("span", { className: "aux" }, "Highest priority first")),
      !items.length ? h("div", { className: "block-body" },
        h("div", { className: "empty" }, "Awaiting the first assessment window.")) :
        h("table", null,
          h("thead", null, h("tr", null,
            h("th", null, "Priority"), h("th", null, "Action"), h("th", null, "Subsystem"),
            h("th", null, "Basis"), h("th", null, "Reference"))),
          h("tbody", null, items.map(function (it, i) {
            if (it.error) return h("tr", { key: i }, h("td", { colSpan: 5 }, it.error));
            return h("tr", { key: i },
              h("td", null, h("span", { className: cls("tag", PRIO_STATE[it.priority]) }, it.priority)),
              h("td", null, it.action),
              h("td", null, it.subsystem),
              h("td", { style: { color: "#4a5260" } }, it.rationale),
              h("td", { className: "mono" }, it.reference)
            );
          }))
        ),
      h("div", { className: "block-body", style: { borderTop: "1px solid #d3d7de" } },
        h("div", { className: "hint" },
          "Actions are drawn from routine piston engine maintenance practice for the four modelled subsystems. " +
          "This is decision support built on a reduced-order model and does not replace the approved aircraft maintenance manual.")
      )
    )
  );
}

function ReportTab(props) {
  const r = props.report;
  if (!r) return h("div", { className: "empty" }, "No sortie in progress. Start a sortie to accumulate a report.");

  const eff = r.efficiency || {};
  const times = r.time_in_recommendation_s || {};
  const totalTime = Object.keys(times).reduce(function (s, k) { return s + times[k]; }, 0);
  const faults = r.fault_window_counts || {};

  return h("div", null,
    h("div", { className: "report-head" },
      h("div", null,
        h("div", { className: "report-title" }, "Mission health report"),
        h("div", { className: "report-sub" },
          r.mission_name + ", required duration " + fmt(r.required_duration_s, 0) + " s")
      ),
      h("div", { style: { textAlign: "right" } },
        h("div", { className: "report-sub" }, "Elapsed " + clockFrom(r.elapsed_s)),
        h("div", { className: "report-sub" }, r.windows_assessed + " windows assessed")
      )
    ),
    h("div", { className: "stat-row" },
      h("div", { className: "stat" },
        h("div", { className: "stat-l" }, "Health at start"),
        h("div", { className: "stat-v" }, r.start_health === null ? "–" : (r.start_health * 100).toFixed(1) + "%")),
      h("div", { className: "stat" },
        h("div", { className: "stat-l" }, "Health at end"),
        h("div", { className: cls("stat-v", "v-" + healthState(r.end_health)) },
          r.end_health === null ? "–" : (r.end_health * 100).toFixed(1) + "%")),
      h("div", { className: "stat" },
        h("div", { className: "stat-l" }, "Minimum health"),
        h("div", { className: cls("stat-v", "v-" + healthState(r.min_health)) },
          r.min_health === null ? "–" : (r.min_health * 100).toFixed(1) + "%")),
      h("div", { className: "stat" },
        h("div", { className: "stat-l" }, "Net change"),
        h("div", { className: cls("stat-v", r.health_change !== null && r.health_change < -0.02 ? "v-caution" : null) },
          r.health_change === null ? "–" : (r.health_change * 100).toFixed(2) + " pts"))
    ),
    h("div", { className: "grid3", style: { marginTop: "12px" } },
      h("div", { className: "block" },
        h("div", { className: "block-head" }, "Findings"),
        h("div", { className: "block-body" },
          h("dl", { className: "kv" },
            h("dt", null, "Dominant finding"),
            h("dd", null, r.dominant_finding === "HEALTHY" ? "No fault attributed"
              : r.dominant_finding.charAt(0) + r.dominant_finding.slice(1).toLowerCase()),
            h("dt", null, "Worst disposition"),
            h("dd", null, r.worst_recommendation ? (REC_TEXT[r.worst_recommendation] || r.worst_recommendation) : "–"),
            h("dt", null, "Anomaly windows"), h("dd", null, r.anomaly_windows + " of " + r.windows_assessed),
            h("dt", null, "Anomaly rate"), h("dd", null, r.anomaly_rate === null ? "–" : (r.anomaly_rate * 100).toFixed(1) + "%"),
            h("dt", null, "Peak anomaly score"), h("dd", null, fmt(r.peak_anomaly_score, 2))
          )
        )
      ),
      h("div", { className: "block" },
        h("div", { className: "block-head" }, "Efficiency over sortie"),
        h("div", { className: "block-body" },
          h("dl", { className: "kv" },
            h("dt", null, "Mean shaft power"), h("dd", null, fmt(eff.mean_power_kw, 3) + " kW"),
            h("dt", null, "Mean power deficit"), h("dd", null, pct(eff.mean_power_deficit_pct, 2)),
            h("dt", null, "Peak power deficit"), h("dd", null, pct(eff.peak_power_deficit_pct, 2)),
            h("dt", null, "Mean fuel penalty"), h("dd", null, pct(eff.mean_bsfc_penalty_pct, 2)),
            h("dt", null, "Peak fuel penalty"), h("dd", null, pct(eff.peak_bsfc_penalty_pct, 2))
          )
        )
      ),
      h("div", { className: "block" },
        h("div", { className: "block-head" }, "Event totals"),
        h("div", { className: "block-body" },
          h("dl", { className: "kv" },
            h("dt", null, "Warning events"), h("dd", null, (r.alert_counts || {}).WARNING || 0),
            h("dt", null, "Caution events"), h("dd", null, (r.alert_counts || {}).CAUTION || 0),
            h("dt", null, "Information events"), h("dd", null, (r.alert_counts || {}).INFO || 0)
          )
        )
      )
    ),
    h("div", { className: "block" },
      h("div", { className: "block-head" }, "Time in disposition"),
      !totalTime ? h("div", { className: "block-body" }, h("div", { className: "empty" }, "Not enough elapsed time to apportion.")) :
        h("table", null,
          h("thead", null, h("tr", null,
            h("th", null, "Disposition"), h("th", { className: "num" }, "Seconds"), h("th", { className: "num" }, "Share"))),
          h("tbody", null, Object.keys(times).map(function (k) {
            return h("tr", { key: k },
              h("td", null, h("span", { className: cls("tag", REC_STATE[k] || "info") }, REC_TEXT[k] || k)),
              h("td", { className: "num" }, fmt(times[k], 1)),
              h("td", { className: "num" }, pct((times[k] / totalTime) * 100, 1))
            );
          }))
        )
    ),
    Object.keys(faults).length ? h("div", { className: "block" },
      h("div", { className: "block-head" }, "Attribution distribution"),
      h("table", null,
        h("thead", null, h("tr", null,
          h("th", null, "Attributed subsystem"), h("th", { className: "num" }, "Windows"), h("th", { className: "num" }, "Share"))),
        h("tbody", null, Object.keys(faults).map(function (k) {
          return h("tr", { key: k },
            h("td", null, k === "HEALTHY" ? "No fault attributed" : k.charAt(0) + k.slice(1).toLowerCase()),
            h("td", { className: "num" }, faults[k]),
            h("td", { className: "num" }, pct((faults[k] / r.windows_assessed) * 100, 1)));
        }))
      )
    ) : null
  );
}

const FAULT_9_CLASSES = [
  { id: 0, name: "NORMAL", label: "Normal Operation", color: "#1b8a5a" },
  { id: 1, name: "MISFIRE", label: "Misfire Conditions", color: "#d9381e" },
  { id: 2, name: "INJECTOR_ABNORMALITY", label: "Injector Abnormalities", color: "#e67e22" },
  { id: 3, name: "CODING_DEGRADATION", label: "Cooling Degradation", color: "#d35400" },
  { id: 4, name: "LUBRICATION_ISSUE", label: "Lubrication Issues", color: "#8e44ad" },
  { id: 5, name: "SENSOR_DRIFT_FAILURE", label: "Sensor Drift / Failure", color: "#2980b9" },
  { id: 6, name: "COMBUSTION_INSTABILITY", label: "Combustion Instability", color: "#c0392b" },
  { id: 7, name: "OVERHEATING", label: "Overheating Trends", color: "#b03a2e" },
  { id: 8, name: "ABNORMAL_VIBRATION", label: "Abnormal Vibration", color: "#7d3c98" },
];

function XGBoostTab(props) {
  const a = props.assessment;
  const xgb = (a && (a.xgboost || a.diagnosis)) || null;
  const probs = (xgb && xgb.probabilities) || {};
  const topDevs = (xgb && xgb.top_deviations) || [];
  const shap = xgb && xgb.shap_explanation;

  return h("div", null,
    h("div", { className: "stat-row" },
      h("div", { className: "stat" },
        h("div", { className: "stat-l" }, "Active ML Classifier"),
        h("div", { className: "stat-v", style: { fontSize: "16px", color: "#1f5fa8" } }, "XGBoost Physics DT"),
        h("div", { className: "stat-n" }, "60 Features (Residuals + Trends)")),
      h("div", { className: "stat" },
        h("div", { className: "stat-l" }, "Diagnostic Attribution"),
        h("div", { className: cls("stat-v", xgb && xgb.predicted_fault !== "NORMAL" && xgb.predicted_fault !== "HEALTHY" ? "v-warning" : "v-normal"), style: { fontSize: "17px" } },
          xgb ? (xgb.predicted_fault || "–").replace(/_/g, " ") : "–"),
        h("div", { className: "stat-n" }, xgb ? "Confidence " + ((xgb.confidence || 0) * 100).toFixed(1) + "%" : "Awaiting window")),
      h("div", { className: "stat" },
        h("div", { className: "stat-l" }, "Decision Margin"),
        h("div", { className: "stat-v" }, xgb && xgb.margin !== undefined ? ((xgb.margin) * 100).toFixed(1) + "%" : "–"),
        h("div", { className: "stat-n" }, xgb ? "Runner-up: " + (xgb.runner_up || "None").replace(/_/g, " ") : "–")),
      h("div", { className: "stat" },
        h("div", { className: "stat-l" }, "Test Macro F1"),
        h("div", { className: "stat-v v-normal" }, "0.9040"),
        h("div", { className: "stat-n" }, "89.70% Accuracy on held-out sorties"))
    ),

    h("div", { className: "grid2", style: { marginTop: "12px" } },
      h("div", null,
        h("div", { className: "block" },
          h("div", { className: "block-head" },
            "9-Class Real-Time Probability Distribution",
            h("span", { className: "aux" }, "Softmax Output Vector")
          ),
          h("div", { className: "block-body" },
            FAULT_9_CLASSES.map(function(clsItem) {
              const p = probs[clsItem.name] !== undefined ? probs[clsItem.name] : (clsItem.name === "NORMAL" && (!xgb || xgb.predicted_fault === "HEALTHY" || xgb.predicted_fault === "NORMAL") ? 1.0 : 0.0);
              const isPred = (xgb && xgb.predicted_fault === clsItem.name) || (clsItem.name === "NORMAL" && (!xgb || xgb.predicted_fault === "HEALTHY" || xgb.predicted_fault === "NORMAL"));
              return h("div", { key: clsItem.name, style: { marginBottom: "11px" } },
                h("div", { style: { display: "flex", justifyContent: "space-between", fontSize: "12px", marginBottom: "3px" } },
                  h("span", { style: { fontWeight: isPred ? "700" : "500", color: isPred ? clsItem.color : "#2c3e50" } },
                    (isPred ? "▶ " : "") + clsItem.label + " (" + clsItem.name + ")"
                  ),
                  h("span", { className: "mono", style: { fontWeight: "700", color: isPred ? clsItem.color : "#606f7b" } },
                    (p * 100).toFixed(1) + "%"
                  )
                ),
                h("div", { className: "meter", style: { height: "9px", background: "#eef1f5" } },
                  h("span", { style: { width: Math.max(1, p * 100) + "%", background: clsItem.color, transition: "width 0.3s ease" } })
                )
              );
            })
          )
        )
      ),

      h("div", null,
        h("div", { className: "block" },
          h("div", { className: "block-head" },
            "Experimental Benchmark (Held-Out Test Sorties)",
            h("span", { className: "aux" }, "Model Comparison")
          ),
          h("div", { className: "block-body" },
            h("table", null,
              h("thead", null,
                h("tr", null,
                  h("th", null, "Model Architecture"),
                  h("th", { className: "num" }, "Features"),
                  h("th", { className: "num" }, "Accuracy"),
                  h("th", { className: "num" }, "Macro F1")
                )
              ),
              h("tbody", null,
                h("tr", null,
                  h("td", null, "Model A: Raw Sensors Only"),
                  h("td", { className: "num" }, "10"),
                  h("td", { className: "num" }, "59.55%"),
                  h("td", { className: "num" }, "0.5574")
                ),
                h("tr", null,
                  h("td", null, "Model B: Sensors + Env + Trends"),
                  h("td", { className: "num" }, "31"),
                  h("td", { className: "num" }, "71.52%"),
                  h("td", { className: "num" }, "0.6981")
                ),
                h("tr", { style: { background: "#e8f4fd", fontWeight: "600" } },
                  h("td", null, "★ Model C: Physics Digital Twin (Active)"),
                  h("td", { className: "num" }, "60"),
                  h("td", { className: "num", style: { color: "#1b8a5a" } }, "89.70%"),
                  h("td", { className: "num", style: { color: "#1b8a5a" } }, "0.9040")
                )
              )
            ),
            h("div", { className: "hint", style: { marginTop: "10px" } },
              "Physics Digital Twin residuals decouple cross-environmental sensor drift from true thermodynamic faults, yielding an empirical +30.15% accuracy gain over raw telemetry alone."
            )
          )
        ),

        h("div", { className: "block", style: { marginTop: "12px" } },
          h("div", { className: "block-head" }, "Live Diagnostic Feature Deviations"),
          h("div", { className: "block-body" },
            !topDevs.length ? h("div", { className: "empty" }, "Awaiting telemetry frames to extract 30s feature window.") :
              h("table", null,
                h("thead", null,
                  h("tr", null,
                    h("th", null, "Indicator / Feature"),
                    h("th", { className: "num" }, "Deviation Value"),
                    h("th", null, "Status")
                  )
                ),
                h("tbody", null,
                  topDevs.map(function(dev, idx) {
                    const isHigh = Math.abs(dev.value) >= 2.0;
                    return h("tr", { key: idx },
                      h("td", { style: { fontWeight: "600" } }, dev.feature),
                      h("td", { className: "num mono" }, dev.value),
                      h("td", null,
                        h("span", { className: cls("tag", isHigh ? "warning" : "normal") },
                          isHigh ? "DEVIATED" : "NOMINAL"
                        )
                      )
                    );
                  })
                )
              )
          )
        )
      )
    ),

    // Explainable AI: TreeSHAP / DeepSHAP Feature Attribution
    h("div", { className: "shap-panel" },
      h("div", { className: "shap-head" },
        h("span", { className: "shap-title" }, "Explainable AI: TreeSHAP Feature Attribution (Exact Game-Theoretic Shapley Values)"),
        h("span", { className: "shap-badge" }, "Explainability Engine")
      ),
      shap ? [
        h("div", { key: "summary", className: "shap-summary-box" },
          h("b", null, "Diagnostic Attribution: "),
          shap.summary || "Attributions nominal.",
          h("span", { style: { marginLeft: "14px", color: "var(--ink-3)", fontFamily: "var(--mono)", fontSize: "11px" } },
            "Base Expected E[f(x)]: " + (shap.base_value !== undefined ? shap.base_value.toFixed(3) : "0.000") +
            " | Output Margin f(x): " + (shap.output_margin !== undefined ? shap.output_margin.toFixed(3) : "0.000")
          )
        ),
        h("div", { key: "bars", className: "shap-bars-grid" },
          // Positive drivers
          h("div", null,
            h("div", { className: "shap-col-title pos" },
              h("span", null, "▲ Risk-Increasing Feature Drivers (Pushing Toward Fault)"),
              h("span", null, "φ > 0")
            ),
            (shap.positive_drivers && shap.positive_drivers.length) ?
              shap.positive_drivers.map(function(item, idx) {
                const absVal = Math.min(1.0, Math.abs(item.shap_value) * 1.5);
                return h("div", { key: idx, className: "shap-bar-item" },
                  h("span", { className: "shap-feat-name", title: item.feature }, item.feature),
                  h("div", { className: "shap-track" },
                    h("div", { className: "shap-fill-pos", style: { width: Math.max(6, absVal * 100) + "%" } })
                  ),
                  h("span", { className: "shap-val-text", style: { color: "var(--warning)" } },
                    "+" + item.shap_value.toFixed(3)
                  )
                );
              }) :
              h("div", { className: "empty" }, "No positive anomaly drivers in current window.")
          ),
          // Negative suppressors
          h("div", null,
            h("div", { className: "shap-col-title neg" },
              h("span", null, "▼ Nominal Envelope Factors (Anchoring Normal Baseline)"),
              h("span", null, "φ < 0")
            ),
            (shap.negative_suppressors && shap.negative_suppressors.length) ?
              shap.negative_suppressors.map(function(item, idx) {
                const absVal = Math.min(1.0, Math.abs(item.shap_value) * 1.5);
                return h("div", { key: idx, className: "shap-bar-item" },
                  h("span", { className: "shap-feat-name", title: item.feature }, item.feature),
                  h("div", { className: "shap-track" },
                    h("div", { className: "shap-fill-neg", style: { width: Math.max(6, absVal * 100) + "%" } })
                  ),
                  h("span", { className: "shap-val-text", style: { color: "var(--accent)" } },
                    item.shap_value.toFixed(3)
                  )
                );
              }) :
              h("div", { className: "empty" }, "No significant negative suppressor attributions.")
          )
        ),
        h("div", { key: "math-note", className: "hint", style: { marginTop: "12px", borderTop: "1px solid var(--border)", paddingTop: "8px" } },
          "TreeSHAP computes exact polynomial-time Shapley values (Lundberg et al.) attributing the contribution of each physics residual and trend feature to the final classification: f(x) = E[f(x)] + Σ φ_i. " +
          "Features with positive φ_i directly pushed the model toward " + (xgb.predicted_fault || "FAULT") + ", while negative φ_i anchored the diagnosis toward healthy nominal operation."
        )
      ] : h("div", { className: "empty" }, "Accumulating feature window to calculate TreeSHAP Shapley attributions...")
    )
  );
}

// -------------------------------------------------------------------- Math Formatting Helper
function MathFrac(num, den) {
  return h("span", { className: "math-frac" },
    h("span", { className: "math-num" }, num),
    h("span", { className: "math-den" }, den)
  );
}

// -------------------------------------------------------------------- DigitalTwinTab
function DigitalTwinTab(props) {
  const { telemetry, expected, physicsState, tick, controls, activeFault, faultComponent, faultSeverity, assessment } = props;
  const t = telemetry || {};
  const exp = expected || {};
  const phys = physicsState || {};
  const ctrl = controls || {};

  // Real-time engine parameters synchronized with actual live data
  const rpm = Number.isFinite(t.rpm) ? t.rpm : 4500;
  const expRpm = Number.isFinite(exp.rpm) ? exp.rpm : rpm;
  const cht = Number.isFinite(t.cht) ? t.cht : 85.0;
  const expCht = Number.isFinite(exp.cht) ? exp.cht : cht;
  const egt = Number.isFinite(t.egt) ? t.egt : 680.0;
  const expEgt = Number.isFinite(exp.egt) ? exp.egt : egt;
  const oilP = Number.isFinite(t.oil_pressure_psi) ? t.oil_pressure_psi * 0.0689476 : 4.1;
  const expOilP = Number.isFinite(exp.oil_pressure_psi) ? exp.oil_pressure_psi * 0.0689476 : oilP;
  const oilT = Number.isFinite(t.oil_temperature) ? t.oil_temperature : 85.0;
  const expOilT = Number.isFinite(exp.oil_temperature) ? exp.oil_temperature : oilT;
  const fuel = Number.isFinite(t.fuel_flow_lph) ? t.fuel_flow_lph : 22.0;
  const expFuel = Number.isFinite(exp.fuel_flow_lph) ? exp.fuel_flow_lph : fuel;

  const thr = Number.isFinite(t.throttle) ? t.throttle : (ctrl.throttle !== undefined ? ctrl.throttle : 0.65);
  const altFt = ctrl.altitude_ft !== undefined ? ctrl.altitude_ft : 0;
  const ambC = ctrl.ambient_c !== undefined ? ctrl.ambient_c : 15.0;

  // Active fault detection for live digital twin visual demonstration
  const effFault = activeFault ||
    (t.fault_type) ||
    (assessment && assessment.fault) ||
    (assessment && assessment.diagnosis && assessment.diagnosis.predicted_fault !== "NORMAL" && assessment.diagnosis.predicted_fault !== "HEALTHY" && assessment.diagnosis.predicted_fault) ||
    null;

  const isCoolingFault = Boolean(effFault && (effFault.includes("COOL") || effFault.includes("TEMP") || effFault.includes("LEAK")));
  const isLubFault = Boolean(effFault && (effFault.includes("OIL") || effFault.includes("LUB")));
  const isMisfireFault = Boolean(effFault && (effFault.includes("MISFIRE") || effFault.includes("CYL")));
  const isBearingFault = Boolean(effFault && (effFault.includes("BEARING") || effFault.includes("VIB")));

  let misfireCyl = 3;
  if (faultComponent) {
    if (faultComponent.includes("1")) misfireCyl = 1;
    else if (faultComponent.includes("2")) misfireCyl = 2;
    else if (faultComponent.includes("3")) misfireCyl = 3;
    else if (faultComponent.includes("4")) misfireCyl = 4;
  }

  const isa = phys.isa || {
    altitude_m: (altFt * 0.3048).toFixed(1),
    t_amb_k: (ambC + 273.15).toFixed(2),
    p_amb_kpa: (101.325 * Math.pow(1.0 - 2.25577e-5 * (altFt * 0.3048), 5.25588)).toFixed(2),
    rho_air: (1.225 * Math.pow(1.0 - 2.25577e-5 * (altFt * 0.3048), 4.25588)).toFixed(4),
    rho_ratio: "1.000",
  };
  const air = phys.air_path || {
    p_man_kpa: (isa.p_amb_kpa * (0.35 + 0.65 * thr)).toFixed(2),
    eta_v: "0.825",
    m_dot_air_kgs: "0.0482",
  };
  const fuelSys = phys.fuel_system || {
    afr_target: "14.70",
    lambda_val: "1.000",
    m_dot_fuel_kgs: "0.00328",
    fuel_flow_calc_lph: fuel.toFixed(2),
  };
  const crank = phys.crankshaft || {
    omega_rads: ((2 * Math.PI * rpm) / 60).toFixed(1),
    t_ind_nm: "148.50",
    t_fric_nm: (11.5 + 0.0038 * rpm + (isBearingFault ? 32.0 : 0.0)).toFixed(2),
    t_load_nm: "128.20",
    dw_dt: "0.000",
    j_inertia: 0.185,
  };
  const therm = phys.thermal || {
    q_in_cht_w: isCoolingFault ? "58200.0" : "42500.0",
    q_cool_cht_w: isCoolingFault ? "21400.0" : "41800.0",
    dcht_dt: isCoolingFault ? "1.450" : "0.015",
    degt_dt: "0.022",
  };
  const lub = phys.lubrication || {
    mu_oil_pa_s: isLubFault ? "0.0082" : "0.0245",
    oil_press_calc_bar: isLubFault ? "1.42" : oilP.toFixed(2),
  };

  // Synchronized crankshaft angle (720 degrees full 4-stroke cycle)
  const crankCycleDeg = ((rpm / 60) * tick * 45) % 720;
  const crankDeg = crankCycleDeg % 360;
  const rad = (crankDeg * Math.PI) / 180;

  // 4-Cylinder kinematic positions
  const cylAngles = [
    crankCycleDeg % 720,
    (crankCycleDeg + 180) % 720,
    (crankCycleDeg + 360) % 720,
    (crankCycleDeg + 540) % 720,
  ];

  // Piston heights (140 to 184 px)
  const pHeights = cylAngles.map(function(ang, idx) {
    const r = (ang * Math.PI) / 180;
    const jitter = (isMisfireFault && idx + 1 === misfireCyl) ? Math.sin(tick * 5) * 3 : 0;
    return 162 - 20 * Math.cos(r) + jitter;
  });

  const cylX = [185, 275, 365, 455];

  // Dynamic EGT glowing gradient
  const egtNorm = Math.min(1.0, Math.max(0.0, (egt - 450) / 450));
  const egtGlow = egtNorm > 0.65 ? "#ef4444" : (egtNorm > 0.35 ? "#f97316" : "#c2410c");

  // Dynamic CHT thermal color
  const chtWarning = isCoolingFault || cht > 115.0;
  const chtColor = chtWarning ? "#ef4444" : (cht > 95.0 ? "#f97316" : "#38bdf8");

  return h("div", null,
    // Top digital twin summary metrics
    h("div", { className: "stat-row" },
      h("div", { className: "stat" },
        h("div", { className: "stat-l" }, "Digital Twin Core"),
        h("div", { className: "stat-v", style: { fontSize: "18px", color: "#1f5fa8" } }, "4-Cyl 4-Stroke MVEM"),
        h("div", { className: "stat-n" }, "Coupled Rotational ODE + Poppet Valve Timing")),
      h("div", { className: "stat" },
        h("div", { className: "stat-l" }, "Throttle / Manifold"),
        h("div", { className: "stat-v" }, (thr * 100).toFixed(1) + "% / " + air.p_man_kpa + " kPa"),
        h("div", { className: "stat-n" }, "Volumetric Eff: " + air.eta_v + " | ṁ_air: " + air.m_dot_air_kgs + " kg/s")),
      h("div", { className: "stat" },
        h("div", { className: "stat-l" }, "Fuel Delivery"),
        h("div", { className: "stat-v" }, fuel.toFixed(1) + " L/h"),
        h("div", { className: "stat-n" }, "Target AFR: " + fuelSys.afr_target + " (λ " + fuelSys.lambda_val + ")")),
      h("div", { className: "stat" },
        h("div", { className: "stat-l" }, "Engine Speed"),
        h("div", { className: "stat-v" }, rpm.toFixed(0) + " RPM"),
        h("div", { className: "stat-n" }, "ω: " + crank.omega_rads + " rad/s | Propeller Synchronized"))
    ),

    h("div", { className: "dt-tab-grid", style: { marginTop: "14px" } },
      // Virtual Engine Schematic SVG
      h("div", { className: "twin-schematic-box" },
        h("div", { className: "twin-schematic-title" },
          h("span", null, "Virtual Engine Digital Twin — Real-Time 4-Stroke Cutaway"),
          h("span", { className: cls("badge", effFault && "warn") },
            effFault ? "⚠️ FAULT DEMO ACTIVE" : "PHYSICS LIVE SYNCHRONIZED"
          )
        ),

        // Prominent Fault Injection Demonstration Banner
        effFault ? h("div", {
          style: {
            background: "rgba(239, 68, 68, 0.15)",
            border: "1px solid rgba(239, 68, 68, 0.45)",
            borderRadius: "4px",
            padding: "8px 14px",
            marginBottom: "12px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            color: "#fca5a5",
          }
        },
          h("span", { style: { fontWeight: "700", fontSize: "13px" } },
            "⚠️ DEMONSTRATING ACTIVE FAULT: " + effFault + (faultSeverity ? " (Severity " + Number(faultSeverity).toFixed(2) + ")" : "")
          ),
          h("span", { style: { fontSize: "11px", background: "#ef4444", color: "#fff", padding: "2px 8px", borderRadius: "3px", fontWeight: "700" } },
            isCoolingFault ? "THERMAL OVERHEAT" : (isLubFault ? "PRESSURE LOSS" : (isMisfireFault ? "CYL " + misfireCyl + " MISFIRE" : "BEARING WEAR"))
          )
        ) : h("div", {
          style: {
            background: "rgba(14, 165, 233, 0.12)",
            border: "1px solid rgba(14, 165, 233, 0.3)",
            borderRadius: "4px",
            padding: "7px 14px",
            marginBottom: "12px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            color: "#7dd3fc",
          }
        },
          h("span", { style: { fontWeight: "600", fontSize: "12.5px" } },
            "● Virtual Digital Twin Synchronized With Real Engine Telemetry"
          ),
          h("span", { style: { fontSize: "11px", color: "#38bdf8", fontWeight: "700" } }, "4-STROKE CYCLE NOMINAL")
        ),

        h("svg", {
          viewBox: "0 0 720 365",
          className: "schematic-svg",
          style: { width: "100%", height: "auto", background: "#090d14", borderRadius: "4px" }
        },
          h("defs", null,
            h("pattern", { id: "schemGrid", width: "20", height: "20", patternUnits: "userSpaceOnUse" },
              h("path", { d: "M 20 0 L 0 0 0 20", fill: "none", stroke: "#161f30", strokeWidth: "0.5" })
            ),
            h("linearGradient", { id: "egtPipe", x1: "0%", y1: "0%", x2: "100%", y2: "0%" },
              h("stop", { offset: "0%", stopColor: "#c2410c" }),
              h("stop", { offset: "100%", stopColor: egtGlow })
            ),
            h("linearGradient", { id: "pistonGrad", x1: "0%", y1: "0%", x2: "0%", y2: "100%" },
              h("stop", { offset: "0%", stopColor: "#64748b" }),
              h("stop", { offset: "100%", stopColor: "#334155" })
            )
          ),
          h("rect", { width: "720", height: "365", fill: "url(#schemGrid)" }),

          // Air Intake with dynamic airflow stream
          h("polygon", { points: "15,65 52,78 52,112 15,125", fill: "#1e293b", stroke: "#38bdf8", strokeWidth: "1.5" }),
          h("text", { x: "20", y: "98", fill: "#38bdf8", fontSize: "9", fontFamily: "var(--mono)", fontWeight: "700" }, "AIR IN"),
          h("path", { d: "M 10 95 L 50 95", stroke: "#38bdf8", strokeWidth: "2", strokeDasharray: "4 3" }),

          // Throttle Body rotating dynamically with actual throttle %
          h("rect", { x: "52", y: "84", width: "42", height: "22", fill: "#1e293b", stroke: "#64748b", strokeWidth: "1.5" }),
          h("line", {
            x1: "73", y1: "85",
            x2: String(73 + 10 * Math.cos(thr * Math.PI * 0.45)),
            y2: String(95 + 10 * Math.sin(thr * Math.PI * 0.45)),
            stroke: "#38bdf8", strokeWidth: "3"
          }),
          h("text", { x: "53", y: "78", fill: "#94a3b8", fontSize: "8", fontFamily: "var(--mono)" }, "THROTTLE " + (thr * 100).toFixed(0) + "%"),

          // Intake Manifold runner
          h("path", {
            d: "M 94 95 L 135 95 L 135 110 L 490 110",
            fill: "none", stroke: "#0284c7", strokeWidth: "6", strokeLinecap: "round"
          }),
          h("text", { x: "155", y: "103", fill: "#7dd3fc", fontSize: "9", fontFamily: "var(--mono)" }, "INTAKE MANIFOLD: " + air.p_man_kpa + " kPa"),

          // Common Rail Fuel Delivery Line
          h("line", { x1: "155", y1: "118", x2: "485", y2: "118", stroke: "#eab308", strokeWidth: "2.5" }),
          h("text", { x: "492", y: "121", fill: "#fde047", fontSize: "8", fontFamily: "var(--mono)" }, "FUEL RAIL (" + fuel.toFixed(1) + " L/h)"),

          // Engine Cast Cylinder Head
          h("rect", {
            x: "150", y: "124", width: "340", height: "18", rx: "3",
            fill: "#1e293b", stroke: "#475569", strokeWidth: "1.5"
          }),
          h("text", { x: "155", y: "136", fill: "#94a3b8", fontSize: "8", fontFamily: "var(--mono)" }, "DOHC CYLINDER HEAD"),

          // Coolant Jacket Surrounding Cylinders
          h("rect", {
            x: "150", y: "142", width: "340", height: "92", rx: "4",
            fill: isCoolingFault ? "rgba(239, 68, 68, 0.22)" : "rgba(14, 165, 233, 0.08)",
            stroke: isCoolingFault ? "#ef4444" : "#0ea5e9",
            strokeWidth: isCoolingFault ? "2.5" : "1",
            strokeDasharray: isCoolingFault ? "6 3" : "none",
          }),
          isCoolingFault ? h("text", { x: "180", y: "139", fill: "#ef4444", fontSize: "9", fontWeight: "800", fontFamily: "var(--mono)" },
            "⚠️ COOLANT LEAK / THERMAL RUNAWAY (CHT: " + cht.toFixed(1) + "°C)"
          ) : null,

          // 4 Cylinders with animated Poppet Valves, Spark Plugs, Pistons & Cycles
          cylX.map(function(cx, idx) {
            const cylNum = idx + 1;
            const py = pHeights[idx];
            const ang = cylAngles[idx];

            // 4-Stroke Cycle determination:
            // 0-180: Power, 180-360: Exhaust, 360-540: Intake, 540-720: Compression
            let phase = "POWER";
            let intakeValveOpen = false;
            let exhaustValveOpen = false;
            let isSparking = false;

            if (ang < 180) {
              phase = "POWER";
              isSparking = ang < 45;
            } else if (ang < 360) {
              phase = "EXHAUST";
              exhaustValveOpen = true;
            } else if (ang < 540) {
              phase = "INTAKE";
              intakeValveOpen = true;
            } else {
              phase = "COMPRESSION";
            }

            const isAfflictedMisfire = isMisfireFault && (cylNum === misfireCyl);
            if (isAfflictedMisfire) {
              isSparking = false;
            }

            const cylCht = isCoolingFault ? (cht + (idx * 3.5)).toFixed(1) : (cht - 2 + idx).toFixed(1);

            return h("g", { key: idx },
              // Cylinder Liner Wall
              h("rect", {
                x: cx - 26, y: 142, width: "52", height: "88",
                fill: "#0f172a",
                stroke: isAfflictedMisfire ? "#ef4444" : (isCoolingFault ? "#f97316" : "#475569"),
                strokeWidth: isAfflictedMisfire ? "2.5" : "2"
              }),

              // Left: Intake Poppet Valve
              h("line", {
                x1: cx - 14, y1: 124,
                x2: cx - 14, y2: intakeValveOpen ? 146 : 142,
                stroke: "#38bdf8", strokeWidth: "2"
              }),
              h("polygon", {
                points: (cx - 19) + "," + (intakeValveOpen ? 146 : 142) + " " + (cx - 9) + "," + (intakeValveOpen ? 146 : 142) + " " + (cx - 14) + "," + (intakeValveOpen ? 149 : 144),
                fill: intakeValveOpen ? "#38bdf8" : "#64748b"
              }),

              // Center: Spark Plug with Ceramic Insulator
              h("rect", { x: cx - 2.5, y: 122, width: "5", height: "12", fill: "#f8fafc" }),
              h("rect", { x: cx - 3.5, y: 130, width: "7", height: "4", fill: "#94a3b8" }),
              h("line", { x1: cx, y1: 134, x2: cx, y2: 142, stroke: "#cbd5e1", strokeWidth: "1.5" }),

              // Right: Exhaust Poppet Valve
              h("line", {
                x1: cx + 14, y1: 124,
                x2: cx + 14, y2: exhaustValveOpen ? 146 : 142,
                stroke: "#fb923c", strokeWidth: "2"
              }),
              h("polygon", {
                points: (cx + 9) + "," + (exhaustValveOpen ? 146 : 142) + " " + (cx + 19) + "," + (exhaustValveOpen ? 146 : 142) + " " + (cx + 14) + "," + (exhaustValveOpen ? 149 : 144),
                fill: exhaustValveOpen ? "#fb923c" : "#64748b"
              }),

              // Combustion Spark / Flame Effect
              isSparking && !isAfflictedMisfire ? h("polygon", {
                points: (cx-18)+",144 "+(cx-6)+",154 "+cx+",146 "+(cx+6)+",156 "+(cx+18)+",144 "+(cx+10)+",150 "+(cx-10)+",150",
                fill: "#f59e0b", opacity: "0.95"
              }) : null,

              // Misfire Warning Callout
              isAfflictedMisfire ? h("g", null,
                h("text", { x: cx - 12, y: "155", fill: "#ef4444", fontSize: "13", fontWeight: "800", fontFamily: "var(--mono)" }, "⚡✕"),
                h("text", { x: cx - 22, y: "168", fill: "#f87171", fontSize: "7", fontWeight: "700", fontFamily: "var(--mono)" }, "MISFIRE")
              ) : null,

              // Reciprocating Piston Head
              h("rect", {
                x: cx - 24, y: py, width: "48", height: "18", rx: "2",
                fill: isAfflictedMisfire ? "#7f1d1d" : "url(#pistonGrad)",
                stroke: isAfflictedMisfire ? "#ef4444" : "#94a3b8",
                strokeWidth: "1.5"
              }),
              // Piston Rings
              h("line", { x1: cx - 22, y1: py + 4, x2: cx + 22, y2: py + 4, stroke: "#475569", strokeWidth: "1" }),
              h("line", { x1: cx - 22, y1: py + 8, x2: cx + 22, y2: py + 8, stroke: "#475569", strokeWidth: "1" }),
              // Gudgeon Wrist Pin
              h("circle", { cx: cx, cy: py + 10, r: "3", fill: "#cbd5e1" }),

              // Connecting Rod
              h("line", {
                x1: cx, y1: py + 10,
                x2: String(cx + 14 * Math.cos(rad + idx * Math.PI * 0.5)),
                y2: String(265 + 14 * Math.sin(rad + idx * Math.PI * 0.5)),
                stroke: isAfflictedMisfire ? "#ef4444" : "#94a3b8",
                strokeWidth: "3.5", strokeLinecap: "round"
              }),

              // Crankpin Journal
              h("circle", {
                cx: String(cx + 14 * Math.cos(rad + idx * Math.PI * 0.5)),
                cy: String(265 + 14 * Math.sin(rad + idx * Math.PI * 0.5)),
                r: "5", fill: isBearingFault && idx === 2 ? "#ef4444" : "#cbd5e1"
              }),

              // Cylinder Number & Live CHT
              h("text", { x: cx - 18, y: "216", fill: chtColor, fontSize: "8", fontFamily: "var(--mono)", fontWeight: "700" },
                "C" + cylNum + " " + cylCht + "°C"
              ),

              // 4-Stroke Phase Tag Badge
              h("rect", {
                x: cx - 26, y: "223", width: "52", height: "13", rx: "2",
                fill: isAfflictedMisfire ? "#450a0a" : (phase === "POWER" ? "#451a03" : "#1e293b"),
                stroke: isAfflictedMisfire ? "#ef4444" : (phase === "POWER" ? "#f59e0b" : "#475569")
              }),
              h("text", {
                x: cx - 22, y: "232",
                fill: isAfflictedMisfire ? "#fca5a5" : (phase === "POWER" ? "#fbbf24" : "#94a3b8"),
                fontSize: "6.5", fontWeight: "700", fontFamily: "var(--mono)"
              },
                isAfflictedMisfire ? "MISFIRE" : phase
              )
            );
          }),

          // Bearing Vibration Shockwaves on Journal #3
          isBearingFault ? h("g", null,
            h("circle", { cx: "365", cy: "265", r: "18", fill: "none", stroke: "#ef4444", strokeWidth: "2", strokeDasharray: "4 2", opacity: "0.9" }),
            h("circle", { cx: "365", cy: "265", r: "28", fill: "none", stroke: "#f97316", strokeWidth: "1.5", strokeDasharray: "6 3", opacity: "0.7" }),
            h("text", { x: "270", y: "254", fill: "#f87171", fontSize: "9", fontWeight: "700", fontFamily: "var(--mono)" },
              "⚠️ JOURNAL BEARING WEAR & HIGH VIBRATION"
            )
          ) : null,

          // Crankshaft Main Beam
          h("line", { x1: "135", y1: "265", x2: "510", y2: "265", stroke: isBearingFault ? "#ef4444" : "#94a3b8", strokeWidth: "5" }),
          h("text", { x: "135", y: "280", fill: "#94a3b8", fontSize: "8.5", fontFamily: "var(--mono)" }, "CRANKSHAFT (J=0.185)"),

          // Flywheel & Output Shaft
          h("circle", { cx: "525", cy: "265", r: "24", fill: "#1e293b", stroke: "#38bdf8", strokeWidth: "3" }),
          h("line", {
            x1: "525", y1: "265",
            x2: String(525 + 22 * Math.cos(rad)),
            y2: String(265 + 22 * Math.sin(rad)),
            stroke: "#38bdf8", strokeWidth: "2.5"
          }),
          h("text", { x: "505", y: "302", fill: "#38bdf8", fontSize: "9", fontWeight: "700", fontFamily: "var(--mono)" }, rpm.toFixed(0) + " RPM"),

          // Propeller Hub & Spinning Blades
          h("rect", { x: "555", y: "260", width: "16", height: "10", fill: "#64748b" }),
          h("polygon", { points: "571,257 590,265 571,273", fill: "#0284c7" }),
          // Propeller spinning motion blur disc
          h("ellipse", { cx: "582", cy: "265", rx: "10", ry: "65", fill: "rgba(56,189,248,0.15)", stroke: "#38bdf8", strokeWidth: "1", strokeDasharray: "4 3" }),
          // Rotating blades
          h("line", {
            x1: String(582 - 55 * Math.sin(rad)),
            y1: String(265 - 55 * Math.cos(rad)),
            x2: String(582 + 55 * Math.sin(rad)),
            y2: String(265 + 55 * Math.cos(rad)),
            stroke: "#94a3b8", strokeWidth: "4.5", strokeLinecap: "round"
          }),
          h("text", { x: "596", y: "270", fill: "#7dd3fc", fontSize: "8", fontFamily: "var(--mono)", fontWeight: "600" }, "PROPELLER"),

          // Exhaust Manifold & Pipe
          h("path", {
            d: "M 185 238 L 185 246 L 455 246 L 500 246 L 500 232 L 670 232",
            fill: "none", stroke: "url(#egtPipe)", strokeWidth: "5.5", strokeLinecap: "round"
          }),
          h("text", { x: "615", y: "224", fill: egtGlow, fontSize: "9", fontWeight: "700", fontFamily: "var(--mono)" }, "EXHAUST: " + egt.toFixed(1) + "°C"),

          // Oil Sump with Submerged Oil Pickup & Strainer
          h("rect", {
            x: "150", y: "295", width: "340", height: "26", rx: "3",
            fill: isLubFault ? "#451a03" : "#172554",
            stroke: isLubFault ? "#ef4444" : "#1d4ed8",
            strokeWidth: isLubFault ? "2" : "1.5"
          }),
          // Oil Pickup Tube dipping into oil sump
          h("path", { d: "M 320 270 L 320 306 L 310 312", fill: "none", stroke: "#94a3b8", strokeWidth: "3" }),
          // Submerged Strainer Mesh
          h("circle", { cx: "310", cy: "312", r: "5", fill: "#334155", stroke: "#64748b", strokeWidth: "1" }),
          h("text", {
            x: "165", y: "312",
            fill: isLubFault ? "#fca5a5" : "#93c5fd",
            fontSize: "8.5", fontWeight: isLubFault ? "700" : "500",
            fontFamily: "var(--mono)"
          },
            isLubFault
              ? "⚠️ OIL PRESSURE DROP: " + oilP.toFixed(2) + " bar — LUBRICATION COLLAPSE"
              : "OIL SUMP: " + oilP.toFixed(2) + " bar (" + (oilP * 14.5038).toFixed(1) + " psi) | " + oilT.toFixed(1) + "°C"
          ),

          // CHT Sensor Marker
          h("circle", { cx: "365", cy: "135", r: "4", fill: chtColor, stroke: "#fff", strokeWidth: "1" }),
          h("text", { x: "375", y: "138", fill: chtColor, fontSize: "8.5", fontWeight: "700", fontFamily: "var(--mono)" }, "CHT: " + cht.toFixed(1) + "°C")
        )
      ),

      // Real-time Parameter Visualization Table
      h("div", { className: "block" },
        h("div", { className: "block-head" },
          "Real-Time Parameter Visualization (Measured vs Digital Twin)",
          h("span", { className: "aux" }, "Residual & Normalized z-score")
        ),
        h("div", { className: "block-body" },
          h("table", { className: "dt-param-table" },
            h("thead", null,
              h("tr", null,
                h("th", null, "Parameter"),
                h("th", { className: "num" }, "Measured (Real)"),
                h("th", { className: "num" }, "Twin (Expected)"),
                h("th", { className: "num" }, "Residual Δ"),
                h("th", { className: "num" }, "Z-Score"),
                h("th", null, "Status")
              )
            ),
            h("tbody", null,
              [
                { name: "Engine Speed (RPM)", meas: rpm.toFixed(0) + " rpm", pred: expRpm.toFixed(0) + " rpm", delta: (rpm - expRpm).toFixed(1), z: ((rpm - expRpm) / 25).toFixed(2), unit: "rpm" },
                { name: "Cylinder Head Temp (CHT)", meas: cht.toFixed(1) + " °C", pred: expCht.toFixed(1) + " °C", delta: (cht - expCht).toFixed(2), z: ((cht - expCht) / 4.0).toFixed(2), unit: "°C" },
                { name: "Exhaust Gas Temp (EGT)", meas: egt.toFixed(1) + " °C", pred: expEgt.toFixed(1) + " °C", delta: (egt - expEgt).toFixed(2), z: ((egt - expEgt) / 15.0).toFixed(2), unit: "°C" },
                { name: "Oil Pressure", meas: oilP.toFixed(2) + " bar", pred: expOilP.toFixed(2) + " bar", delta: (oilP - expOilP).toFixed(3), z: ((oilP - expOilP) / 0.15).toFixed(2), unit: "bar" },
                { name: "Oil Temperature", meas: oilT.toFixed(1) + " °C", pred: expOilT.toFixed(1) + " °C", delta: (oilT - expOilT).toFixed(2), z: ((oilT - expOilT) / 2.0).toFixed(2), unit: "°C" },
                { name: "Fuel Flow Rate", meas: fuel.toFixed(2) + " L/h", pred: expFuel.toFixed(2) + " L/h", delta: (fuel - expFuel).toFixed(2), z: ((fuel - expFuel) / 0.5).toFixed(2), unit: "L/h" },
                { name: "Manifold Pressure", meas: air.p_man_kpa + " kPa", pred: air.p_man_kpa + " kPa", delta: "0.00", z: "0.00", unit: "kPa" },
                { name: "Air Mass Flow Rate", meas: air.m_dot_air_kgs + " kg/s", pred: air.m_dot_air_kgs + " kg/s", delta: "0.000", z: "0.00", unit: "kg/s" },
              ].map(function(row, idx) {
                const zVal = parseFloat(row.z);
                const isWarn = Math.abs(zVal) >= 3.0;
                const isCaut = Math.abs(zVal) >= 2.0;
                const badgeCls = isWarn ? "warning" : (isCaut ? "caution" : "normal");
                const badgeTxt = isWarn ? "WARNING" : (isCaut ? "CAUTION" : "NORMAL");

                return h("tr", { key: idx },
                  h("td", { style: { fontWeight: "600" } }, row.name),
                  h("td", { className: "num" }, row.meas),
                  h("td", { className: "num", style: { color: "var(--accent)" } }, row.pred),
                  h("td", { className: "num" }, row.delta),
                  h("td", { className: "num " + (isWarn ? "z-warn" : (isCaut ? "z-caut" : "z-ok")) }, row.z),
                  h("td", null, h("span", { className: "tag " + badgeCls }, badgeTxt))
                );
              })
            )
          )
        )
      )
    ),

    // Governing Physics Equations Section — Formatted Mathematically
    h("div", { className: "block", style: { marginTop: "16px" } },
      h("div", { className: "block-head" },
        "Governing Physics Equations (Digital Twin CORE Mechanics & Thermodynamic Equations)",
        h("span", { className: "aux" }, "Evaluated in Real-Time at 10 Hz")
      ),
      h("div", { className: "block-body" },
        h("div", { className: "equations-container" },
          // Eq 1: ISA Atmosphere
          h("div", { className: "eq-card" },
            h("div", { className: "eq-head" },
              h("span", { className: "eq-title" }, "1. International Standard Atmosphere (ISA) Environmental Model"),
              h("span", { className: "eq-category" }, "Atmosphere & Density")
            ),
            h("div", { className: "eq-desc" },
              "Models ambient air pressure and density lapse rates as a function of UAV altitude, establishing baseline aerodynamic charging for the engine intake."
            ),
            h("div", { className: "eq-math" },
              h("div", { className: "math-formula" },
                h("span", { className: "math-var" }, "P"),
                h("sub", { className: "math-sub" }, "a"),
                h("span", null, "(h) = "),
                h("span", { className: "math-var" }, "P"),
                h("sub", { className: "math-sub" }, "0"),
                h("span", { className: "math-sym" }, "·"),
                h("span", { className: "math-paren" }, "["),
                h("span", null, "1 − "),
                MathFrac(
                  h("span", null, h("span", { className: "math-var" }, "L"), " · ", h("span", { className: "math-var" }, "h")),
                  h("span", null, h("span", { className: "math-var" }, "T"), h("sub", { className: "math-sub" }, "0"))
                ),
                h("span", { className: "math-paren" }, "]"),
                h("sup", { className: "math-sup" },
                  MathFrac(h("span", { className: "math-var" }, "g"), h("span", null, h("span", { className: "math-var" }, "R"), " · ", h("span", { className: "math-var" }, "L")))
                ),
                h("span", { style: { margin: "0 16px", color: "var(--border-2)" } }, "|"),
                h("span", { className: "math-var" }, "ρ"),
                h("sub", { className: "math-sub" }, "a"),
                h("span", null, " = "),
                MathFrac(
                  h("span", null, h("span", { className: "math-var" }, "P"), h("sub", { className: "math-sub" }, "a")),
                  h("span", null, h("span", { className: "math-var" }, "R"), " · ", h("span", { className: "math-var" }, "T"), h("sub", { className: "math-sub" }, "a"))
                ),
                h("span", { style: { margin: "0 16px", color: "var(--border-2)" } }, "|"),
                h("span", { className: "math-var" }, "σ"),
                h("span", null, " = "),
                MathFrac(
                  h("span", null, h("span", { className: "math-var" }, "ρ"), h("sub", { className: "math-sub" }, "a")),
                  h("span", null, h("span", { className: "math-var" }, "ρ"), h("sub", { className: "math-sub" }, "0"))
                )
              )
            ),
            h("div", { className: "eq-live-terms" },
              h("span", { className: "eq-term-chip" }, "Alt h: ", h("b", null, isa.altitude_m + " m (" + altFt + " ft)")),
              h("span", { className: "eq-term-chip" }, "P_amb: ", h("b", null, isa.p_amb_kpa + " kPa")),
              h("span", { className: "eq-term-chip" }, "T_amb: ", h("b", null, ambC + " °C (" + isa.t_amb_k + " K)")),
              h("span", { className: "eq-term-chip" }, "Air Density ρ: ", h("b", null, isa.rho_air + " kg/m³")),
              h("span", { className: "eq-term-chip" }, "Density Ratio σ: ", h("b", null, isa.rho_ratio))
            )
          ),

          // Eq 2: Air Path & Manifold Dynamics
          h("div", { className: "eq-card" },
            h("div", { className: "eq-head" },
              h("span", { className: "eq-title" }, "2. Intake Manifold Filling Dynamics & Volumetric Efficiency"),
              h("span", { className: "eq-category" }, "Air Path Dynamics")
            ),
            h("div", { className: "eq-desc" },
              "Continuity differential equation governing intake manifold pressure response to rapid throttle butterfly transients and engine displacement pumping."
            ),
            h("div", { className: "eq-math" },
              h("div", { className: "math-formula" },
                MathFrac(h("span", null, "d", h("span", { className: "math-var" }, "P"), h("sub", { className: "math-sub" }, "man")), "dt"),
                h("span", null, " = "),
                MathFrac(
                  h("span", null, h("span", { className: "math-var" }, "R"), " · ", h("span", { className: "math-var" }, "T"), h("sub", { className: "math-sub" }, "man")),
                  h("span", null, h("span", { className: "math-var" }, "V"), h("sub", { className: "math-sub" }, "man"))
                ),
                h("span", null, " · (ṁ", h("sub", { className: "math-sub" }, "thr"), " − ṁ", h("sub", { className: "math-sub" }, "cyl"), ")"),
                h("span", { style: { margin: "0 16px", color: "var(--border-2)" } }, "|"),
                h("span", null, "ṁ", h("sub", { className: "math-sub" }, "air"), " = "),
                h("span", { className: "math-var" }, "η"),
                h("sub", { className: "math-sub" }, "v"),
                h("span", null, " · "),
                h("span", { className: "math-paren" }, "("),
                MathFrac(
                  h("span", null, h("span", { className: "math-var" }, "V"), h("sub", { className: "math-sub" }, "d"), " · ", h("span", { className: "math-var" }, "N")),
                  "120"
                ),
                h("span", { className: "math-paren" }, ")"),
                h("span", null, " · "),
                h("span", { className: "math-var" }, "ρ"),
                h("sub", { className: "math-sub" }, "man")
              )
            ),
            h("div", { className: "eq-live-terms" },
              h("span", { className: "eq-term-chip" }, "P_man: ", h("b", null, air.p_man_kpa + " kPa")),
              h("span", { className: "eq-term-chip" }, "Volumetric Eff η_v: ", h("b", null, air.eta_v)),
              h("span", { className: "eq-term-chip" }, "Air Mass Flow ṁ_air: ", h("b", null, air.m_dot_air_kgs + " kg/s")),
              h("span", { className: "eq-term-chip" }, "Engine Displ V_d: ", h("b", null, "2.40 L"))
            )
          ),

          // Eq 3: Fuel Injection & Stoichiometry
          h("div", { className: "eq-card" },
            h("div", { className: "eq-head" },
              h("span", { className: "eq-title" }, "3. Electronic Fuel Injection (EFI) & Chemical Energy Release"),
              h("span", { className: "eq-category" }, "Fuel System & Combustion")
            ),
            h("div", { className: "eq-desc" },
              "Calculates required fuel mass flow for stoichiometric or target air/fuel ratio (AFR) and computes chemical energy release rate into the combustion chamber."
            ),
            h("div", { className: "eq-math" },
              h("div", { className: "math-formula" },
                h("span", null, "ṁ", h("sub", { className: "math-sub" }, "fuel"), " = "),
                MathFrac(h("span", null, "ṁ", h("sub", { className: "math-sub" }, "air")), h("span", null, "AFR", h("sub", { className: "math-sub" }, "target"))),
                h("span", { style: { margin: "0 16px", color: "var(--border-2)" } }, "|"),
                h("span", { className: "math-var" }, "λ"),
                h("span", null, " = "),
                MathFrac("AFR", "14.70"),
                h("span", { style: { margin: "0 16px", color: "var(--border-2)" } }, "|"),
                h("span", null, "Q̇", h("sub", { className: "math-sub" }, "comb"), " = ṁ", h("sub", { className: "math-sub" }, "fuel"), " · LHV · "),
                h("span", { className: "math-var" }, "η"),
                h("sub", { className: "math-sub" }, "thermal")
              )
            ),
            h("div", { className: "eq-live-terms" },
              h("span", { className: "eq-term-chip" }, "Target AFR: ", h("b", null, fuelSys.afr_target)),
              h("span", { className: "eq-term-chip" }, "Lambda λ: ", h("b", null, fuelSys.lambda_val)),
              h("span", { className: "eq-term-chip" }, "Fuel Mass Flow ṁ_f: ", h("b", null, fuelSys.m_dot_fuel_kgs + " kg/s")),
              h("span", { className: "eq-term-chip" }, "Fuel Flow Volume: ", h("b", null, fuelSys.fuel_flow_calc_lph + " L/h"))
            )
          ),

          // Eq 4: Crankshaft Rotational Dynamics (ODE)
          h("div", { className: "eq-card" },
            h("div", { className: "eq-head" },
              h("span", { className: "eq-title" }, "4. Crankshaft Rotational Dynamics (Newton-Euler ODE)"),
              h("span", { className: "eq-category" }, "Mechanical Drivetrain")
            ),
            h("div", { className: "eq-desc" },
              "First-order rotational dynamic torque balance between combustion indicated work, hydrodynamic friction losses, and aerodynamic propeller load torque."
            ),
            h("div", { className: "eq-math" },
              h("div", { className: "math-formula" },
                h("span", { className: "math-var" }, "J"),
                h("span", null, " · "),
                MathFrac(h("span", { className: "math-var" }, "dω"), "dt"),
                h("span", null, " = "),
                h("span", { className: "math-var" }, "T"),
                h("sub", { className: "math-sub" }, "ind"),
                h("span", null, " − "),
                h("span", { className: "math-var" }, "T"),
                h("sub", { className: "math-sub" }, "fric"),
                h("span", null, " − "),
                h("span", { className: "math-var" }, "T"),
                h("sub", { className: "math-sub" }, "load"),
                h("span", { style: { margin: "0 16px", color: "var(--border-2)" } }, "|"),
                h("span", { className: "math-var" }, "N"),
                h("span", null, " = "),
                MathFrac(h("span", null, "60 · ", h("span", { className: "math-var" }, "ω")), h("span", null, "2π"))
              )
            ),
            h("div", { className: "eq-live-terms" },
              h("span", { className: "eq-term-chip" }, "Inertia J: ", h("b", null, crank.j_inertia + " kg·m²")),
              h("span", { className: "eq-term-chip" }, "Angular Vel ω: ", h("b", null, crank.omega_rads + " rad/s")),
              h("span", { className: "eq-term-chip" }, "Indicated Torque T_ind: ", h("b", null, crank.t_ind_nm + " N·m")),
              h("span", { className: "eq-term-chip" }, "Friction Torque T_fric: ", h("b", null, crank.t_fric_nm + " N·m")),
              h("span", { className: "eq-term-chip" }, "Load Torque T_load: ", h("b", null, crank.t_load_nm + " N·m")),
              h("span", { className: "eq-term-chip" }, "dω/dt: ", h("b", null, crank.dw_dt + " rad/s²"))
            )
          ),

          // Eq 5: Cylinder Head Thermal Balance
          h("div", { className: "eq-card" },
            h("div", { className: "eq-head" },
              h("span", { className: "eq-title" }, "5. Cylinder Head Temperature (CHT) Lumped Thermal Lag"),
              h("span", { className: "eq-category" }, "Thermal Thermodynamics")
            ),
            h("div", { className: "eq-desc" },
              "First-order thermal capacity differential equation modeling heat transfer from combustion gases into the aluminium head and heat removal via coolant circuit."
            ),
            h("div", { className: "eq-math" },
              h("div", { className: "math-formula" },
                h("span", { className: "math-var" }, "τ"),
                h("sub", { className: "math-sub" }, "cht"),
                h("span", null, " · "),
                MathFrac(h("span", null, "d(CHT)"), "dt"),
                h("span", null, " = Q̇", h("sub", { className: "math-sub" }, "comb,cyl"), " − Q̇", h("sub", { className: "math-sub" }, "cool"), " · (CHT − T", h("sub", { className: "math-sub" }, "amb"), ")")
              )
            ),
            h("div", { className: "eq-live-terms" },
              h("span", { className: "eq-term-chip" }, "Time Const τ_cht: ", h("b", null, "22.0 s")),
              h("span", { className: "eq-term-chip" }, "Combustion Heat In: ", h("b", null, therm.q_in_cht_w + " W")),
              h("span", { className: "eq-term-chip" }, "Cooling Heat Out: ", h("b", null, therm.q_cool_cht_w + " W")),
              h("span", { className: "eq-term-chip" }, "d(CHT)/dt: ", h("b", null, therm.dcht_dt + " °C/s"))
            )
          ),

          // Eq 6: Exhaust Gas Thermal Balance
          h("div", { className: "eq-card" },
            h("div", { className: "eq-head" },
              h("span", { className: "eq-title" }, "6. Exhaust Gas Temperature (EGT) Enthalpy Balance"),
              h("span", { className: "eq-category" }, "Thermal Exhaust")
            ),
            h("div", { className: "eq-desc" },
              "Predicts exhaust runner gas temperature and thermocouple response time under varying fuel-air mixtures and combustion flame propagation speeds."
            ),
            h("div", { className: "eq-math" },
              h("div", { className: "math-formula" },
                h("span", { className: "math-var" }, "τ"),
                h("sub", { className: "math-sub" }, "egt"),
                h("span", null, " · "),
                MathFrac(h("span", null, "d(EGT)"), "dt"),
                h("span", null, " = Q̇", h("sub", { className: "math-sub" }, "comb,exh"), " − ṁ", h("sub", { className: "math-sub" }, "air"), " · c", h("sub", { className: "math-sub" }, "p"), " · (EGT − T", h("sub", { className: "math-sub" }, "amb"), ")")
              )
            ),
            h("div", { className: "eq-live-terms" },
              h("span", { className: "eq-term-chip" }, "Time Const τ_egt: ", h("b", null, "2.5 s")),
              h("span", { className: "eq-term-chip" }, "EGT Gradient: ", h("b", null, therm.degt_dt + " °C/s")),
              h("span", { className: "eq-term-chip" }, "Exhaust Enthalpy: ", h("b", null, (fuel * 44000 * 0.36 / 3.6).toFixed(0) + " W"))
            )
          ),

          // Eq 7: Hydrodynamic Lubrication
          h("div", { className: "eq-card" },
            h("div", { className: "eq-head" },
              h("span", { className: "eq-title" }, "7. Hydrodynamic Lubrication & Petroff Viscous Resistance"),
              h("span", { className: "eq-category" }, "Tribology & Lubrication")
            ),
            h("div", { className: "eq-desc" },
              "Coupled Reynolds lubrication equation relating oil gallery delivery pressure, oil pump rotational speed, and temperature-dependent Andrade dynamic viscosity."
            ),
            h("div", { className: "eq-math" },
              h("div", { className: "math-formula" },
                h("span", { className: "math-var" }, "P"),
                h("sub", { className: "math-sub" }, "oil"),
                h("span", null, " = K", h("sub", { className: "math-sub" }, "pump"), " · N · μ(T", h("sub", { className: "math-sub" }, "oil"), ")"),
                h("span", { style: { margin: "0 16px", color: "var(--border-2)" } }, "|"),
                h("span", { className: "math-var" }, "μ"),
                h("span", null, "(T", h("sub", { className: "math-sub" }, "oil"), ") = μ", h("sub", { className: "math-sub" }, "0"), " · exp"),
                h("span", { className: "math-paren" }, "["),
                MathFrac(
                  h("span", null, h("span", { className: "math-var" }, "b")),
                  h("span", null, "T", h("sub", { className: "math-sub" }, "oil"), " + 273.15")
                ),
                h("span", null, " − c"),
                h("span", { className: "math-paren" }, "]")
              )
            ),
            h("div", { className: "eq-live-terms" },
              h("span", { className: "eq-term-chip" }, "Dynamic Viscosity μ: ", h("b", null, lub.mu_oil_pa_s + " Pa·s")),
              h("span", { className: "eq-term-chip" }, "Calculated P_oil: ", h("b", null, lub.oil_press_calc_bar + " bar")),
              h("span", { className: "eq-term-chip" }, "Measured P_oil: ", h("b", null, oilP.toFixed(2) + " bar"))
            )
          ),

          // Eq 8: Analytical Redundancy Residual Vector
          h("div", { className: "eq-card" },
            h("div", { className: "eq-head" },
              h("span", { className: "eq-title" }, "8. Analytical Redundancy Residuals & Normalized Z-Scores"),
              h("span", { className: "eq-category" }, "Fault Detection & Isolation")
            ),
            h("div", { className: "eq-desc" },
              "Decouples nominal operating condition variations from true physical component degradations by evaluating zero-mean normalized statistical residuals."
            ),
            h("div", { className: "eq-math" },
              h("div", { className: "math-formula" },
                h("span", { className: "math-var" }, "r"),
                h("sub", { className: "math-sub" }, "i"),
                h("span", null, " = Y", h("sub", { className: "math-sub" }, "meas,i"), " − Y", h("sub", { className: "math-sub" }, "pred,i"), " (Twin)"),
                h("span", { style: { margin: "0 20px", color: "var(--border-2)" } }, "|"),
                h("span", { className: "math-var" }, "z"),
                h("sub", { className: "math-sub" }, "i"),
                h("span", null, " = "),
                MathFrac(h("span", null, h("span", { className: "math-var" }, "r"), h("sub", { className: "math-sub" }, "i")), h("span", null, "σ", h("sub", { className: "math-sub" }, "noise,i")))
              )
            ),
            h("div", { className: "eq-live-terms" },
              h("span", { className: "eq-term-chip" }, "r_rpm: ", h("b", null, (rpm - expRpm).toFixed(1) + " rpm")),
              h("span", { className: "eq-term-chip" }, "r_cht: ", h("b", null, (cht - expCht).toFixed(2) + " °C")),
              h("span", { className: "eq-term-chip" }, "r_egt: ", h("b", null, (egt - expEgt).toFixed(2) + " °C")),
              h("span", { className: "eq-term-chip" }, "r_oil_p: ", h("b", null, (oilP - expOilP).toFixed(3) + " bar"))
            )
          )
        )
      )
    )
  );
}

// -------------------------------------------------------------------- app

const TABS = [
  ["monitoring", "Telemetry"],
  ["twin_physics", "Digital Twin & Physics"],
  ["xgboost", "⚡ ML Diagnostics (TreeSHAP)"],
  ["efficiency", "Efficiency"],
  ["alerts", "Fault alerts"],
  ["maintenance", "Maintenance advisory"],
  ["report", "Mission report"],
];

function App() {
  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [telemetry, setTelemetry] = useState(null);
  const [assessment, setAssessment] = useState(null);
  const [efficiency, setEfficiency] = useState(null);
  const [effSummary, setEffSummary] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [alertCounts, setAlertCounts] = useState({});
  const [report, setReport] = useState(null);
  const [status, setStatus] = useState(null);
  const [tab, setTab] = useState("monitoring");
  const [busy, setBusy] = useState(false);
  const [missionS, setMissionS] = useState(600);
  const [banner, setBanner] = useState(null);

  // Hero & Interactive Tutorial state
  const [heroVisible, setHeroVisible] = useState(true);
  const [tutorialActive, setTutorialActive] = useState(false);
  const [tutorialStep, setTutorialStep] = useState(0);

  // Audio alert state (continuous urgent avionics buzzer when fault detected)
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [isFaultActive, setIsFaultActive] = useState(false);
  const buzzerTimerRef = useRef(null);

  // Subsystems & Physics state from backend
  const [subsystems, setSubsystems] = useState(null);
  const [physicsState, setPhysicsState] = useState(null);

  // Sortie live log dialogue
  const [logOpen, setLogOpen] = useState(false);
  const logRows = useRef([]);
  const [logTick, setLogTick] = useState(0);

  // Live manual controls
  const [ctrlAuto,     setCtrlAuto]     = useState(true);
  const [ctrlThrottle, setCtrlThrottle] = useState(0.65);
  const [ctrlAlt,      setCtrlAlt]      = useState(0);
  const [ctrlAmb,      setCtrlAmb]      = useState(15.0);
  // Throttle send rate-limiter (send only when stable for 80ms)
  const ctrlDebounce = useRef(null);
  const wsRef        = useRef(null);  // live ref to the WS for direct sends (fast path)

  // Fault injection panel state
  const [faultType,       setFaultType]       = useState("CYLINDER");
  const [faultSeverity,   setFaultSeverity]   = useState(0.8);
  const [faultTrajectory, setFaultTrajectory] = useState("CONSTANT");
  const [faultComponent,  setFaultComponent]  = useState("");  // blank = default
  const [faultBusy,       setFaultBusy]       = useState(false);
  const [faultStatus,     setFaultStatus]     = useState(null);  // last inject result msg
  const [activeFault,     setActiveFault]     = useState(null);  // currently injected fault name

  // Continuous periodic avionics beep alert while fault is present and audio enabled
  useEffect(function() {
    if (isFaultActive && audioEnabled) {
      if (window.PratibimbAudio) {
        window.PratibimbAudio.setMuted(false);
        window.PratibimbAudio.startContinuousBeep(720);
      }
    } else {
      if (window.PratibimbAudio) {
        window.PratibimbAudio.stopContinuousBeep();
      }
    }
    return function() {
      if (window.PratibimbAudio) {
        window.PratibimbAudio.stopContinuousBeep();
      }
    };
  }, [isFaultActive, audioEnabled]);

  const hist = useRef({
    rpm: [], rpm_exp: [], cht: [], cht_exp: [], egt: [], egt_exp: [],
    oil: [], oil_exp: [], vib: [], vib_exp: [],
    pw: [], pw_exp: [], pdef: [], bsfc: [], bsfc_exp: [], bpen: [], fuel: [],
  });
  const [tick, setTick] = useState(0);

  const push = useCallback(function (key, v) {
    const arr = hist.current[key];
    arr.push(Number.isFinite(v) ? v : NaN);
    if (arr.length > MAX_POINTS) arr.shift();
  }, []);

  const clearHist = useCallback(function () {
    Object.keys(hist.current).forEach(function (k) { hist.current[k].length = 0; });
  }, []);

  useEffect(function () {
    let ws, retry;
    function connect() {
      ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = function () { setConnected(true); ws.send("hello"); };
      ws.onmessage = function (ev) {
        let msg;
        try { msg = JSON.parse(ev.data); } catch (e) { return; }
        if (msg.type === "error") { setBanner(msg.message); return; }
        if (msg.type === "mission_stopped") {
          resetAllStats();
          return;
        }
        if (msg.type !== "telemetry") return;

        if (msg.subsystems) setSubsystems(msg.subsystems);
        if (msg.physics_equations) setPhysicsState(msg.physics_equations);

        // Continuous urgent avionics buzzer trigger on active fault / anomaly
        const hasFault = Boolean(
          activeFault ||
          (msg.assessment && (msg.assessment.anomaly || (msg.assessment.diagnosis && msg.assessment.diagnosis.predicted_fault !== "NORMAL" && msg.assessment.diagnosis.predicted_fault !== "HEALTHY"))) ||
          (msg.xgboost && msg.xgboost.is_anomaly) ||
          (msg.telemetry && msg.telemetry.fault_type)
        );
        setIsFaultActive(hasFault);

        const t = msg.telemetry;
        const exp = (msg.twin && msg.twin.expected) || {};
        const eff = msg.efficiency || null;
        // Sync slider positions from server-authoritative controls (so auto-mode animates them)
        if (msg.controls) {
          setCtrlAuto(msg.controls.auto);
          setCtrlThrottle(msg.controls.throttle);
          setCtrlAlt(msg.controls.altitude_ft);
          setCtrlAmb(msg.controls.ambient_c);
        }
        if (t) {
          setTelemetry(t);
          push("rpm", t.rpm); push("rpm_exp", exp.rpm);
          push("cht", t.cht); push("cht_exp", exp.cht);
          push("egt", t.egt); push("egt_exp", exp.egt);
          push("oil", t.oil_pressure_psi); push("oil_exp", exp.oil_pressure_psi);
          push("vib", t.vibration); push("vib_exp", exp.vibration);

          // Append to sortie live log (real vs DT, using AeroTwin twin expected as DT)
          const liveAlt = (msg.controls && msg.controls.altitude_ft != null) ? msg.controls.altitude_ft : 0;
          const rOilP = t.oil_pressure_psi != null ? t.oil_pressure_psi * 0.0689476 : null;
          const dOilP = exp.oil_pressure_psi != null ? exp.oil_pressure_psi * 0.0689476 : null;
          logRows.current.push({
            t:   t.simulation_time,
            thr: t.throttle,
            alt: liveAlt,
            real: {
              rpm:          t.rpm,
              cht:          t.cht,
              egt:          t.egt,
              oil_pressure: rOilP,
              oil_temperature: t.oil_temperature,
              fuel_flow_lph: t.fuel_flow_lph,
            },
            dt: {
              rpm:          exp.rpm,
              cht:          exp.cht,
              egt:          exp.egt,
              oil_pressure: dOilP,
              oil_temperature: exp.oil_temperature,
              fuel_flow_lph: exp.fuel_flow_lph,
            },
            z: {
              rpm:      exp.rpm      && t.rpm      ? (t.rpm      - exp.rpm)      / 25   : null,
              cht:      exp.cht      && t.cht      ? (t.cht      - exp.cht)      / 4    : null,
              egt:      exp.egt      && t.egt      ? (t.egt      - exp.egt)      / 15   : null,
              oil_press:rOilP != null && dOilP != null ? (rOilP - dOilP) / 0.15 : null,
              oil_temp: exp.oil_temperature && t.oil_temperature ? (t.oil_temperature - exp.oil_temperature) / 2.0 : null,
              fuel_flow:exp.fuel_flow_lph   && t.fuel_flow_lph   ? (t.fuel_flow_lph   - exp.fuel_flow_lph)   / 0.5 : null,
            },
          });
          if (logRows.current.length > 6000) logRows.current.shift();
          setLogTick(function(n){ return n+1; });
        }
        if (eff) {
          setEfficiency(eff);
          push("pw", eff.power_kw); push("pw_exp", eff.power_expected_kw);
          push("pdef", eff.power_deficit_pct);
          push("bsfc", eff.bsfc); push("bsfc_exp", eff.bsfc_expected);
          push("bpen", eff.bsfc_penalty_pct);
          push("fuel", eff.fuel_flow_lph);
        }
        if (msg.assessment) setAssessment(msg.assessment);
        setTick(function (n) { return n + 1; });
      };
      ws.onclose = function () { setConnected(false); retry = setTimeout(connect, 1500); };
      ws.onerror = function () { ws.close(); };
    }
    connect();
    return function () {
      clearTimeout(retry);
      if (ws) { ws.onclose = null; ws.close(); }
    };
  }, [push]);

  // Slower poll for the panels that do not need frame-rate updates.
  useEffect(function () {
    let stop = false;
    async function tickPoll() {
      try {
        const st = await (await fetch("/api/status")).json();
        if (stop) return;
        setStatus(st);
        setRunning(!!st.running);
        if (st.running) {
          const [al, ef, rp] = await Promise.all([
            fetch("/api/alerts?limit=80").then(function (r) { return r.json(); }),
            fetch("/api/efficiency?limit=1").then(function (r) { return r.json(); }),
            fetch("/api/mission/report").then(function (r) { return r.ok ? r.json() : null; }),
          ]);
          if (stop) return;
          setAlerts(al.alerts || []);
          setAlertCounts(al.counts || {});
          setEffSummary(ef.summary || null);
          setReport(rp);
        }
      } catch (e) { /* the connection indicator already reports this */ }
    }
    tickPoll();
    const id = setInterval(tickPoll, 2000);
    return function () { stop = true; clearInterval(id); };
  }, []);

  async function post(path, body) {
    setBusy(true); setBanner(null);
    try {
      const r = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
      });
      if (!r.ok) {
        const d = await r.json().catch(function () { return {}; });
        setBanner(d.detail || ("Request failed with status " + r.status));
      }
      return r.ok;
    } catch (e) {
      setBanner(String(e)); return false;
    } finally { setBusy(false); }
  }

  // Debounced controls sender — avoids flooding the server when dragging sliders
  function sendControls(patch) {
    if (ctrlDebounce.current) clearTimeout(ctrlDebounce.current);
    ctrlDebounce.current = setTimeout(function () {
      fetch("/api/sim/controls", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
    }, 80);
  }


  function resetAllStats() {
    setRunning(false);
    setTelemetry(null);
    setAssessment(null);
    setEfficiency(null);
    setEffSummary(null);
    setAlerts([]);
    setAlertCounts({});
    setReport(null);
    clearHist();
    logRows.current = [];
    setActiveFault(null);
    setFaultStatus(null);
    setIsFaultActive(false);
    setPhysicsState(null);
    setLogOpen(false);
    if (window.PratibimbAudio) {
      window.PratibimbAudio.stopContinuousBeep();
    }
  }

  function start() {
    resetAllStats();
    setRunning(true);
    setLogOpen(true);
    setCtrlAuto(true);   // new sortie always starts in auto mode
    post("/api/sim/start", { seed: 42, mission_duration_s: Number(missionS) });
  }

  async function stop() {
    setBusy(true);
    try {
      await post("/api/sim/stop");
      resetAllStats();
    } catch (e) {
      console.error("[Pratibimb] Stop error:", e);
      resetAllStats();
    } finally {
      setBusy(false);
    }
  }

  async function inject(faultTypeOverride) {
    // Inject fault into the ONGOING sortie — no reset of mission state.
    const ft = faultTypeOverride || faultType;
    setFaultBusy(true); setFaultStatus(null);
    try {
      const body = {
        fault_type: ft,
        severity:   Number(faultSeverity),
        trajectory: faultTrajectory,
        ramp_duration_s: 30,
      };
      if (faultComponent && ft !== "CLEAR") body.component = faultComponent;
      const r = await fetch("/api/sim/inject_fault", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const d = await r.json().catch(function(){ return {}; });
      if (r.ok) {
        if (ft === "CLEAR") {
          setActiveFault(null);
          setIsFaultActive(false);
          setFaultStatus("✅ Fault cleared — engine back to healthy baseline.");
        } else {
          setActiveFault(ft);
          setIsFaultActive(true);
          const tStr = d.injected_at_sim_time != null ? " @ T+" + d.injected_at_sim_time.toFixed(0) + "s" : "";
          setFaultStatus("⚠️ " + ft + " fault injected" + tStr + " (sev " + Number(faultSeverity).toFixed(2) + ", " + faultTrajectory + ")");
        }
      } else {
        setFaultStatus("❌ " + (d.detail || "Injection failed"));
      }
    } catch(e) {
      setFaultStatus("❌ " + String(e));
    } finally {
      setFaultBusy(false);
    }
  }

  const t = telemetry || {};
  const models = (status && status.pipeline && status.pipeline.models_loaded) || {};
  const windowReady = status && status.pipeline && status.pipeline.window_ready;
  const modelsUp = ["anomaly", "diagnosis", "rul"].filter(function (m) { return models[m]; }).length;

  return h("div", { className: "app" },
    // Apple-style natural blur hero overlay
    heroVisible ? h(HeroOverlay, {
      onStartDemo: function() {
        setHeroVisible(false);
        setTutorialActive(true);
        setTutorialStep(0);
      },
      onSkip: function() {
        setHeroVisible(false);
      }
    }) : null,

    // Interactive skippable step-by-step tutorial overlay
    tutorialActive ? h(TutorialOverlay, {
      step: tutorialStep,
      onNext: function() { setTutorialStep(function(s) { return Math.min(TUTORIAL_STEPS.length - 1, s + 1); }); },
      onPrev: function() { setTutorialStep(function(s) { return Math.max(0, s - 1); }); },
      onSkip: function() { setTutorialActive(false); },
      onClose: function() { setTutorialActive(false); }
    }) : null,

    h("div", { className: "masthead" },
      h("div", { className: "ident" },
        h("span", { className: "sysname" }, "PRATIBIMB"),
        h("span", { className: "sysdesc" }, "Physics-Informed Digital Twin for Health Monitoring & Predictive Maintenance of MALE UAVs")
      ),
      h("div", { className: "ident-meta" },
        h("span", null, "Engine ", h("b", null, t.engine_id || "PRATIBIMB-UAV-001")),
        h("span", null, "Sortie time ", h("b", null, clockFrom(t.simulation_time))),
        h("span", null, "Analytics ", h("b", null, modelsUp + " of 3")),
        h("span", { className: "conn" },
          h("span", { className: cls("led", connected ? "live" : "down") }),
          connected ? "Telemetry link established" : "Telemetry link down")
      )
    ),

    // Subsystem Connectivity & Hardware Strip (ECU, FADEC, EDGE, TELEMETRY, GCS)
    h(SubsystemStrip, {
      subsystems: subsystems,
      running: running,
      audioEnabled: audioEnabled,
      onToggleAudio: function() {
        const next = !audioEnabled;
        setAudioEnabled(next);
        if (window.PratibimbAudio) {
          window.PratibimbAudio.setMuted(!next);
        }
      },
      isBuzzerActive: (isFaultActive && audioEnabled),
    }),


    h(SortieLogModal, {
      open:    logOpen,
      onClose: function(){ setLogOpen(false); },
      rows:    logRows.current,
    }),

    h("div", { className: "toolbar" },
      h("div", { className: "tgroup" },
        h("span", { className: "tlabel" }, "Sortie"),
        h("span", { className: "brow" },
          h("button", { className: "go", onClick: start, disabled: busy }, "Start sortie"),
          h("button", { onClick: function () { setLogOpen(true); }, disabled: !running, title: "Open live log" }, "📋 Log"),
          h("button", { onClick: stop, disabled: busy || !running, title: "End sortie and reset all statistics" }, "Stop")
        )
      ),
      h("div", { className: "tgroup" },
        h("span", { className: "tlabel" }, "Required duration"),
        h("input", {
          type: "number", min: 1, value: missionS,
          onChange: function (e) { setMissionS(e.target.value); },
        })
      ),
      h("div", { className: "tgroup" },
        h("span", { className: "tlabel" }, "Simulation"),
        h("span", { style: { fontSize: "12px", fontWeight: 600 } },
          h("span", { className: cls("led", running ? "live" : ""), style: { display: "inline-block", marginRight: "7px" } }),
          running ? "Running" : "Stopped")
      )
    ),

    // ---- Controls panel ----
    running ? h("div", { className: "ctrl-panel" },

      // Auto toggle
      h("div", { className: "ctrl-group" },
        h("div", { className: "ctrl-label" }, "Mode"),
        h("button", {
          className: cls("ctrl-auto-btn", ctrlAuto && "active"),
          onClick: function () {
            const next = !ctrlAuto;
            setCtrlAuto(next);
            sendControls({ auto: next });
          },
          title: ctrlAuto ? "Click to take manual control" : "Click to enable auto flight profile",
        },
          h("span", { className: cls("led", ctrlAuto ? "live" : ""), style: { display: "inline-block", marginRight: "6px" } }),
          ctrlAuto ? "AUTO" : "MANUAL"
        )
      ),

      // Throttle slider
      h("div", { className: "ctrl-group ctrl-wide" },
        h("div", { className: "ctrl-label" },
          "Throttle",
          h("span", { className: "ctrl-val" }, (ctrlThrottle * 100).toFixed(1) + " %")
        ),
        h("input", {
          type: "range", className: "ctrl-slider",
          min: 0, max: 1, step: 0.01,
          value: ctrlThrottle,
          disabled: ctrlAuto,
          onChange: function (e) {
            const v = parseFloat(e.target.value);
            setCtrlThrottle(v);
            sendControls({ throttle: v, auto: false });
          },
        })
      ),

      // Altitude slider
      h("div", { className: "ctrl-group ctrl-wide" },
        h("div", { className: "ctrl-label" },
          "Altitude",
          h("span", { className: "ctrl-val" }, Math.round(ctrlAlt).toLocaleString() + " ft")
        ),
        h("input", {
          type: "range", className: "ctrl-slider",
          min: 0, max: 20000, step: 100,
          value: ctrlAlt,
          disabled: ctrlAuto,
          onChange: function (e) {
            const v = parseFloat(e.target.value);
            setCtrlAlt(v);
            sendControls({ altitude_ft: v, auto: false });
          },
        })
      ),

      // Ambient temp
      h("div", { className: "ctrl-group" },
        h("div", { className: "ctrl-label" },
          "OAT",
          h("span", { className: "ctrl-val" }, ctrlAmb.toFixed(1) + " \u00b0C")
        ),
        h("input", {
          type: "range", className: "ctrl-slider",
          min: -30, max: 50, step: 0.5,
          value: ctrlAmb,
          disabled: ctrlAuto,
          onChange: function (e) {
            const v = parseFloat(e.target.value);
            setCtrlAmb(v);
            sendControls({ ambient_c: v, auto: false });
          },
        })
      )

    ) : null,

    // ---- Fault Injection Panel (live, no mission reset) ----
    running ? h("div", { className: "fault-panel" },
      h("div", { className: "fault-panel-head" },
        h("span", { className: "fault-panel-title" }, "⚡ Live Fault Injection"),
        activeFault
          ? h("span", { className: "fault-badge active" }, "● " + activeFault + " ACTIVE")
          : h("span", { className: "fault-badge clear" }, "● ENGINE HEALTHY")
      ),
      h("div", { className: "fault-panel-body" },

        // Fault type buttons
        h("div", { className: "fault-row" },
          h("span", { className: "fault-sublabel" }, "Fault Type"),
          h("div", { className: "fault-type-btns" },
            [
              { id: "CYLINDER",    icon: "🔴", label: "Cylinder Misfire" },
              { id: "BEARING",     icon: "🔵", label: "Bearing Wear" },
              { id: "COOLING",     icon: "🟠", label: "Cooling Failure" },
              { id: "LUBRICATION", icon: "🟡", label: "Lubrication Issue" },
            ].map(function(f) {
              return h("button", {
                key: f.id,
                className: cls("fault-type-btn", faultType === f.id && "selected"),
                onClick: function() { setFaultType(f.id); setFaultComponent(""); },
                disabled: faultBusy,
                title: f.label,
              }, f.icon + " " + f.label);
            })
          )
        ),

        // Component picker (cylinder-specific)
        faultType === "CYLINDER" ? h("div", { className: "fault-row" },
          h("span", { className: "fault-sublabel" }, "Target Cylinder"),
          h("div", { className: "fault-type-btns" },
            ["", "CYLINDER_1", "CYLINDER_2", "CYLINDER_3", "CYLINDER_4"].map(function(c) {
              const lbl = c === "" ? "Any (Cyl 3)" : "Cyl " + c.slice(-1);
              return h("button", {
                key: c,
                className: cls("fault-type-btn", "small", faultComponent === c && "selected"),
                onClick: function() { setFaultComponent(c); },
                disabled: faultBusy,
              }, lbl);
            })
          )
        ) : null,

        // Severity + Trajectory row
        h("div", { className: "fault-row fault-row-split" },
          h("div", { className: "fault-col" },
            h("div", { className: "fault-sublabel" },
              "Severity",
              h("span", { className: "fault-sublabel-val" }, " " + Number(faultSeverity).toFixed(2))
            ),
            h("input", {
              type: "range", className: "ctrl-slider",
              min: 0.1, max: 1.0, step: 0.05,
              value: faultSeverity,
              disabled: faultBusy,
              onChange: function(e) { setFaultSeverity(parseFloat(e.target.value)); },
            })
          ),
          h("div", { className: "fault-col" },
            h("div", { className: "fault-sublabel" }, "Trajectory"),
            h("select", {
              className: "fault-select",
              value: faultTrajectory,
              disabled: faultBusy,
              onChange: function(e) { setFaultTrajectory(e.target.value); },
            },
              ["CONSTANT", "LINEAR", "STEP", "EXPONENTIAL"].map(function(t) {
                return h("option", { key: t, value: t }, t);
              })
            )
          )
        ),

        // Action buttons + status
        h("div", { className: "fault-row fault-actions" },
          h("button", {
            className: "fault-inject-btn",
            onClick: function() { inject(); },
            disabled: faultBusy || !running,
            title: "Inject fault into the live running sortie (no restart)",
          }, faultBusy ? "Injecting…" : "⚡ Inject Fault Now"),
          h("button", {
            className: "fault-clear-btn",
            onClick: function() { inject("CLEAR"); },
            disabled: faultBusy || !running || !activeFault,
            title: "Clear active fault and restore healthy baseline",
          }, "✅ Clear Fault"),
          faultStatus ? h("span", { className: "fault-status" }, faultStatus) : null
        )
      )
    ) : null,

    banner ? h("div", { className: "banner" }, banner) : null,
    running && !windowReady ? h("div", { className: "info-bar" },
      "Accumulating the first five second assessment window. Health, attribution and disposition appear once it is complete.") : null,

    h(StatusStrip, { assessment: assessment }),

    h("div", { className: "tabs" },
      TABS.map(function (entry) {
        return h("button", {
          key: entry[0],
          className: cls("tab", tab === entry[0] && "active"),
          onClick: function () { setTab(entry[0]); },
        }, entry[1]);
      })
    ),

    h("div", { className: "panelwrap" },
      tab === "monitoring" ? h(MonitoringTab, { hist: hist.current, telemetry: t, tick: tick, assessment: assessment }) :
      tab === "twin_physics" ? h(DigitalTwinTab, {
        telemetry: t,
        expected: (assessment && assessment.expected) || {},
        physicsState: physicsState,
        tick: tick,
        controls: { throttle: ctrlThrottle, altitude_ft: ctrlAlt, ambient_c: ctrlAmb },
        activeFault: activeFault,
        faultComponent: faultComponent,
        faultSeverity: faultSeverity,
        assessment: assessment,
      }) :
      tab === "xgboost" ? h(XGBoostTab, { assessment: assessment, status: status }) :
      tab === "efficiency" ? h(EfficiencyTab, { hist: hist.current, tick: tick, efficiency: efficiency, effSummary: effSummary }) :
      tab === "alerts" ? h(AlertsTab, { alerts: alerts, counts: alertCounts }) :
      tab === "maintenance" ? h(MaintenanceTab, { items: assessment && assessment.maintenance, assessment: assessment }) :
      h(ReportTab, { report: report })
    ),

    h("div", { className: "foot" },
      "Reduced-order phenomenological engine model. Results are self-consistent within the model equations and have not been " +
      "validated against engine test cell data. Dispatch thresholds and maintenance intervals shown here are demonstration " +
      "defaults and are not certified airworthiness limits.")
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(h(App));
