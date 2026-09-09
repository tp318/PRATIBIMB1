"""
AeroTwin-4 Operating State Extractor.

Parses raw EngineTelemetry frame objects or dictionary data, extracts operating inputs,
and isolates ground-truth fields to guarantee zero leakage.
"""

from typing import Union, Dict, Any
from simulator.telemetry.schema import EngineTelemetry
from .models import OperatingState


class OperatingStateExtractor:
    """
    Extracts operating state inputs from raw telemetry without leaking ground truth.
    """

    @staticmethod
    def extract(telemetry: Union[EngineTelemetry, Dict[str, Any]], ambient_temperature: float = 25.0) -> OperatingState:
        """
        Extract operating state from telemetry object or dictionary.
        """
        if isinstance(telemetry, EngineTelemetry):
            data = telemetry.to_dict()
        else:
            data = telemetry

        # Strict Ground-Truth Isolation Check:
        # Guarantee ground truth fields (severity, degradation_type, etc.) are NOT passed
        return OperatingState(
            timestamp=float(data.get("timestamp", 0.0)),
            simulation_time=float(data.get("simulation_time", 0.0)),
            engine_id=str(data.get("engine_id", "AEROTWIN-4-001")),
            operating_mode=str(data.get("operating_mode", "CRUISE")),
            throttle=float(data.get("throttle", 0.0)),
            rpm=float(data.get("rpm", 0.0)),
            crank_angle=float(data.get("crank_angle", 0.0)),
            ambient_temperature=float(ambient_temperature),
        )
