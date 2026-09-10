"""
frontend.py
===========
Streamlit interactive dashboard for the UAV piston engine digital twin.

Connects to the FastAPI WebSocket backend (`ws://localhost:8000/ws`),
sends pilot control inputs (throttle, altitude, ambient temperature, fault
mode + severity), and streams back real-time telemetry which is plotted
live with Plotly.

Real-time strategy in Streamlit
--------------------------------
Streamlit re-executes the whole script top-to-bottom on every interaction.
To get a smooth ~10 Hz "live" feed without a separate always-on process,
this app uses the classic Streamlit real-time pattern:

    1. While `st.session_state.simulation_active` is True, each script run
       performs exactly ONE control-send + telemetry-receive round trip
       against the WebSocket, appends the result to rolling `deque`
       buffers, redraws the charts, and then calls `st.rerun()` to
       immediately re-execute the script and do the next tick.
    2. The Start/Stop buttons flip `simulation_active` via their
       `on_click` callbacks. Because a widget interaction (e.g. clicking
       Stop, or moving a slider) causes Streamlit to cancel the currently
       running script and start a fresh one, the loop above naturally
       "notices" slider/button changes on the very next tick -- keeping
       UI responsiveness within ~1-2 simulation steps.

Run with:
    streamlit run frontend.py
"""

from __future__ import annotations

import json
import time
from collections import deque

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import websockets
from websockets.sync.client import connect as ws_connect

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------

st.set_page_config(page_title="UAV Piston Engine Digital Twin", layout="wide")

WS_URL = "ws://localhost:8000/ws"
BUFFER_LEN = 300  # 300 samples @ 10 Hz = 30 seconds of rolling history

# ----------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------
# All persistent objects (WebSocket connection, rolling data buffers,
# control values, run flag) must live in st.session_state so they survive
# Streamlit's script reruns.

def _init_state() -> None:
    defaults = {
        "simulation_active": False,
        "ws": None,
        "ws_error": None,
        "t_buf": deque(maxlen=BUFFER_LEN),
        "rpm_exp_buf": deque(maxlen=BUFFER_LEN),
        "rpm_act_buf": deque(maxlen=BUFFER_LEN),
        "cht_exp_buf": deque(maxlen=BUFFER_LEN),
        "cht_act_buf": deque(maxlen=BUFFER_LEN),
        "egt_residual_buf": deque(maxlen=BUFFER_LEN),
        "last_cht_residual": 0.0,
        "throttle": 0.5,
        "altitude_ft": 3000,
        "ambient_c": 15,
        "fault_id": 0,
        "severity": 0.5,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()

FAULT_OPTIONS = {
    0: "0: Healthy",
    1: "1: Misfire",
    2: "2: Cooling Degradation",
    3: "3: Lubrication",
    4: "4: Sensor Drift",
    5: "5: Combustion Instability",
}


# ----------------------------------------------------------------------
# WebSocket helpers
# ----------------------------------------------------------------------


def get_connection():
    """Return a live WebSocket connection, opening a new one if needed."""
    if st.session_state.ws is None:
        st.session_state.ws = ws_connect(WS_URL, open_timeout=5)
        st.session_state.ws_error = None
    return st.session_state.ws


def close_connection() -> None:
    if st.session_state.ws is not None:
        try:
            st.session_state.ws.close()
        except Exception:
            pass
        st.session_state.ws = None


def current_controls_payload(command: str | None = None) -> dict:
    """Build the JSON control message from the current slider/dropdown
    state, optionally attaching a lifecycle command (start/stop/reset)."""
    payload = {
        "throttle": float(st.session_state.throttle),
        "altitude_ft": float(st.session_state.altitude_ft),
        "ambient_c": float(st.session_state.ambient_c),
        "fault_id": int(st.session_state.fault_id),
        "severity": float(st.session_state.severity),
    }
    if command is not None:
        payload["command"] = command
    return payload


def reset_buffers() -> None:
    st.session_state.t_buf.clear()
    st.session_state.rpm_exp_buf.clear()
    st.session_state.rpm_act_buf.clear()
    st.session_state.cht_exp_buf.clear()
    st.session_state.cht_act_buf.clear()
    st.session_state.egt_residual_buf.clear()
    st.session_state.last_cht_residual = 0.0


# ----------------------------------------------------------------------
# Button callbacks
# ----------------------------------------------------------------------


def on_start_clicked() -> None:
    try:
        ws = get_connection()
        ws.send(json.dumps(current_controls_payload(command="start")))
        st.session_state.simulation_active = True
        st.session_state.ws_error = None
    except Exception as exc:  # noqa: BLE001 - surface any connection issue
        st.session_state.ws_error = f"Could not connect to backend: {exc}"
        st.session_state.simulation_active = False


def on_stop_clicked() -> None:
    st.session_state.simulation_active = False
    if st.session_state.ws is not None:
        try:
            st.session_state.ws.send(json.dumps(current_controls_payload(command="stop")))
        except Exception:
            pass
    close_connection()


def on_reset_clicked() -> None:
    if st.session_state.ws is not None:
        try:
            st.session_state.ws.send(json.dumps(current_controls_payload(command="reset")))
        except Exception:
            pass
    reset_buffers()


# ----------------------------------------------------------------------
# Sidebar: pilot controls
# ----------------------------------------------------------------------

with st.sidebar:
    st.header("Pilot Controls")

    st.slider(
        "Throttle", min_value=0.0, max_value=1.0, step=0.01, key="throttle"
    )
    st.slider(
        "Altitude (ft)", min_value=0, max_value=20000, step=100, key="altitude_ft"
    )
    st.slider(
        "Ambient Temperature (°C)", min_value=-10, max_value=50, step=1, key="ambient_c"
    )

    st.divider()
    st.subheader("Fault Injection")

    st.selectbox(
        "Fault Mode",
        options=list(FAULT_OPTIONS.keys()),
        format_func=lambda k: FAULT_OPTIONS[k],
        key="fault_id",
    )
    st.slider(
        "Fault Severity", min_value=0.0, max_value=1.0, step=0.05, key="severity"
    )

    st.divider()
    col_start, col_stop = st.columns(2)
    col_start.button(
        "▶ Start Simulation",
        on_click=on_start_clicked,
        disabled=st.session_state.simulation_active,
        use_container_width=True,
    )
    col_stop.button(
        "■ Stop Simulation",
        on_click=on_stop_clicked,
        disabled=not st.session_state.simulation_active,
        use_container_width=True,
    )
    st.button("↺ Reset", on_click=on_reset_clicked, use_container_width=True)

    if st.session_state.ws_error:
        st.error(st.session_state.ws_error)


# ----------------------------------------------------------------------
# Main area layout
# ----------------------------------------------------------------------

st.title("MALE UAV Aero Piston Engine — Digital Twin")

status_placeholder = st.empty()
chart1_placeholder = st.empty()
chart2_placeholder = st.empty()
chart3_placeholder = st.empty()


def render_status(cht_residual: float) -> None:
    """Render the health alert box based on the CHT residual thresholds."""
    abs_res = abs(cht_residual)
    if abs_res <= 5.0:
        status_placeholder.success(
            f"🟢 Healthy Engine — CHT residual {cht_residual:+.1f} °C"
        )
    elif abs_res <= 15.0:
        status_placeholder.warning(
            f"🟡 Warning — CHT residual {cht_residual:+.1f} °C"
        )
    else:
        status_placeholder.error(
            f"🔴 Critical — CHT residual {cht_residual:+.1f} °C"
        )


def render_charts() -> None:
    t = list(st.session_state.t_buf)

    # ---- Chart 1: RPM actual vs expected ----
    fig_rpm = go.Figure()
    fig_rpm.add_trace(go.Scatter(x=t, y=list(st.session_state.rpm_exp_buf),
                                  mode="lines", name="Expected RPM",
                                  line=dict(color="green")))
    fig_rpm.add_trace(go.Scatter(x=t, y=list(st.session_state.rpm_act_buf),
                                  mode="lines", name="Actual RPM",
                                  line=dict(color="red")))
    fig_rpm.update_layout(title="RPM — Actual vs Expected", height=300,
                           xaxis_title="Time (s)", yaxis_title="RPM",
                           margin=dict(l=40, r=20, t=40, b=30),
                           legend=dict(orientation="h"))
    chart1_placeholder.plotly_chart(fig_rpm, use_container_width=True, key=f"rpm_{len(t)}")

    # ---- Chart 2: CHT actual vs expected ----
    fig_cht = go.Figure()
    fig_cht.add_trace(go.Scatter(x=t, y=list(st.session_state.cht_exp_buf),
                                  mode="lines", name="Expected CHT",
                                  line=dict(color="green")))
    fig_cht.add_trace(go.Scatter(x=t, y=list(st.session_state.cht_act_buf),
                                  mode="lines", name="Actual CHT",
                                  line=dict(color="red")))
    fig_cht.update_layout(title="CHT — Actual vs Expected", height=300,
                           xaxis_title="Time (s)", yaxis_title="CHT (°C)",
                           margin=dict(l=40, r=20, t=40, b=30),
                           legend=dict(orientation="h"))
    chart2_placeholder.plotly_chart(fig_cht, use_container_width=True, key=f"cht_{len(t)}")

    # ---- Chart 3: EGT residual ----
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=t, y=list(st.session_state.egt_residual_buf),
                                  mode="lines", name="EGT Residual",
                                  line=dict(color="blue")))
    if t:
        fig_res.add_shape(type="line", x0=t[0], x1=t[-1], y0=0, y1=0,
                           line=dict(color="gray", dash="dash"))
    fig_res.update_layout(title="EGT Residual (Actual − Expected)", height=300,
                           xaxis_title="Time (s)", yaxis_title="ΔEGT (°C)",
                           margin=dict(l=40, r=20, t=40, b=30),
                           legend=dict(orientation="h"))
    chart3_placeholder.plotly_chart(fig_res, use_container_width=True, key=f"egt_{len(t)}")


# ----------------------------------------------------------------------
# Real-time simulation loop (one tick per script rerun)
# ----------------------------------------------------------------------

if st.session_state.simulation_active:
    loop_start = time.perf_counter()
    try:
        ws = get_connection()

        # Push the latest slider/dropdown values on every tick so control
        # changes reach the backend within 1-2 simulation steps.
        ws.send(json.dumps(current_controls_payload()))

        # Block briefly for the next telemetry frame (backend ticks at
        # 10 Hz, so ~0.2s timeout gives comfortable headroom).
        raw = ws.recv(timeout=0.2)
        frame = json.loads(raw)

        expected = frame["expected"]
        actual = frame["actual"]
        residual = frame["residual"]

        st.session_state.t_buf.append(frame.get("sim_time", time.time()))
        st.session_state.rpm_exp_buf.append(expected["RPM"])
        st.session_state.rpm_act_buf.append(actual["RPM"])
        st.session_state.cht_exp_buf.append(expected["CHT"])
        st.session_state.cht_act_buf.append(actual["CHT"])
        st.session_state.egt_residual_buf.append(residual["EGT"])
        st.session_state.last_cht_residual = residual["CHT"]
        st.session_state.ws_error = None

    except (websockets.exceptions.ConnectionClosed, TimeoutError, OSError) as exc:
        st.session_state.ws_error = f"WebSocket error: {exc}"
        close_connection()
        st.session_state.simulation_active = False
    except Exception as exc:  # noqa: BLE001
        st.session_state.ws_error = f"Unexpected error: {exc}"

    render_status(st.session_state.last_cht_residual)
    render_charts()

    # Pace reruns to roughly match the backend's 10 Hz cadence.
    elapsed = time.perf_counter() - loop_start
    time.sleep(max(0.0, 0.1 - elapsed))
    st.rerun()

else:
    render_status(st.session_state.last_cht_residual)
    render_charts()
    st.info("Press **Start Simulation** in the sidebar to begin streaming telemetry.")
