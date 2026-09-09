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

  const diagState = diag ? (diag.predicted_fault === "HEALTHY" ? "normal" : "warning") : "idle";
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
      h("div", { className: "cell-label" }, "Fault attribution"),
      h("div", { className: cls("cell-value", "sm", "v-" + diagState) },
        diag ? (diag.predicted_fault === "HEALTHY" ? "No fault" : diag.predicted_fault.charAt(0) + diag.predicted_fault.slice(1).toLowerCase()) : "–"),
      h("div", { className: "meter" }, h("span", { style: { width: (diag ? diag.confidence * 100 : 0) + "%" } })),
      h("div", { className: "cell-note" },
        diag ? "Confidence " + (diag.confidence * 100).toFixed(0) + "%, runner-up " + (diag.runner_up || "").toLowerCase() : "Awaiting first window")
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

// -------------------------------------------------------------------- app

const TABS = [
  ["monitoring", "Monitoring"],
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
      ws.onopen = function () { setConnected(true); ws.send("hello"); };
      ws.onmessage = function (ev) {
        let msg;
        try { msg = JSON.parse(ev.data); } catch (e) { return; }
        if (msg.type === "error") { setBanner(msg.message); return; }
        if (msg.type !== "telemetry") return;

        const t = msg.telemetry;
        const exp = (msg.twin && msg.twin.expected) || {};
        const eff = msg.efficiency || null;
        if (t) {
          setTelemetry(t);
          push("rpm", t.rpm); push("rpm_exp", exp.rpm);
          push("cht", t.cht); push("cht_exp", exp.cht);
          push("egt", t.egt); push("egt_exp", exp.egt);
          push("oil", t.oil_pressure_psi); push("oil_exp", exp.oil_pressure_psi);
          push("vib", t.vibration); push("vib_exp", exp.vibration);
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

  function start() {
    setAssessment(null); setReport(null); setAlerts([]); clearHist();
    post("/api/sim/start", { seed: 42, mission_duration_s: Number(missionS) });
  }

  function inject(fault) {
    setAssessment(null); setReport(null); setAlerts([]); clearHist();
    post("/api/sim/inject_fault", {
      fault_type: fault, severity: 0.8, trajectory: "CONSTANT",
      mission_duration_s: Number(missionS),
    });
  }

  const t = telemetry || {};
  const models = (status && status.pipeline && status.pipeline.models_loaded) || {};
  const windowReady = status && status.pipeline && status.pipeline.window_ready;
  const modelsUp = ["anomaly", "diagnosis", "rul"].filter(function (m) { return models[m]; }).length;

  return h("div", { className: "app" },
    h("div", { className: "masthead" },
      h("div", { className: "ident" },
        h("span", { className: "sysname" }, "AeroTwin-4"),
        h("span", { className: "sysdesc" }, "Engine Health Monitoring System")
      ),
      h("div", { className: "ident-meta" },
        h("span", null, "Engine ", h("b", null, t.engine_id || "AEROTWIN-4-001")),
        h("span", null, "Sortie time ", h("b", null, clockFrom(t.simulation_time))),
        h("span", null, "Analytics ", h("b", null, modelsUp + " of 3")),
        h("span", { className: "conn" },
          h("span", { className: cls("led", connected ? "live" : "down") }),
          connected ? "Telemetry link established" : "Telemetry link down")
      )
    ),

    h("div", { className: "toolbar" },
      h("div", { className: "tgroup" },
        h("span", { className: "tlabel" }, "Sortie"),
        h("span", { className: "brow" },
          h("button", { className: "go", onClick: start, disabled: busy }, "Start sortie"),
          h("button", { onClick: function () { post("/api/sim/stop"); }, disabled: busy || !running }, "Stop")
        )
      ),
      h("div", { className: "tgroup" },
        h("span", { className: "tlabel" }, "Required duration"),
        h("input", {
          type: "number", min: 1, value: missionS,
          onChange: function (e) { setMissionS(e.target.value); },
        })
      ),
      h("div", { className: "tgroup", style: { flex: 1 } },
        h("span", { className: "tlabel" }, "Inject degradation at severity 0.80"),
        h("span", { className: "brow" },
          ["CYLINDER", "BEARING", "COOLING", "LUBRICATION"].map(function (f) {
            return h("button", { key: f, onClick: function () { inject(f); }, disabled: busy },
              f.charAt(0) + f.slice(1).toLowerCase());
          })
        )
      ),
      h("div", { className: "tgroup" },
        h("span", { className: "tlabel" }, "Simulation"),
        h("span", { style: { fontSize: "12px", fontWeight: 600 } },
          h("span", { className: cls("led", running ? "live" : ""), style: { display: "inline-block", marginRight: "7px" } }),
          running ? "Running" : "Stopped")
      )
    ),

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
