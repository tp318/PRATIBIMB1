"""
run_dashboard.py
================
One-command launcher for the AeroTwin-4 operator dashboard.

This script starts the FastAPI server that serves both the REST/WebSocket
API and the HTML dashboard at http://localhost:8001

Usage:
    python run_dashboard.py

Then open: http://localhost:8001

The server streams live engine telemetry via WebSocket (/ws/telemetry) and
exposes the full assessment pipeline (anomaly detection, fault diagnosis, RUL,
mission risk) via REST endpoints documented at http://localhost:8001/docs
"""

import os
import sys

# Ensure the DASHBOARD AND DATA directory is on sys.path so the
# AeroTwin package can be imported regardless of how this script is invoked.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import uvicorn  # noqa: E402 — imported after path fix

if __name__ == "__main__":
    print("=" * 70)
    print("  PROJECT PRATIBIMB: Physics-Informed Digital Twin for MALE UAVs")
    print("=" * 70)
    print()
    print("  Dashboard : http://localhost:8001")
    print("  API docs  : http://localhost:8001/docs")
    print("  WebSocket : ws://localhost:8001/ws/telemetry")
    print()
    print("  After the server starts:")
    print("  1. Open http://localhost:8001 in your browser")
    print("  2. Experience the Apple-style typewriter hero & tutorial")
    print("  3. Click 'Start simulation' to begin the engine run")
    print("  4. View the Virtual Engine Digital Twin & Governing Equations Tab")
    print("  5. Explore Explainable AI (TreeSHAP) attributions & audio fault alerts")
    print()
    print("  Press Ctrl+C to stop.")
    print()

    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(
        "AeroTwin.api.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # reload=True breaks the background simulation task
    )
