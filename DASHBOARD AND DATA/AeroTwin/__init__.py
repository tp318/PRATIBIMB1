"""
AeroTwin-4 - physics-informed Digital Twin for a representative aero piston engine.

Modules inside this package import each other with bare names (``from simulator.runner
import EngineRunner``) rather than package-qualified ones. That only resolves when the
AeroTwin directory itself is on sys.path, which until now every entry point had to
arrange by hand - and which silently broke any importer that did not, such as
``uvicorn AeroTwin.api.server:app``.

Doing the bootstrap here means importing AeroTwin is enough, from anywhere.
"""

import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_PKG_DIR)

for _p in (_PKG_DIR, os.path.join(_PKG_DIR, "phase1"), _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__version__ = "1.0.0"
