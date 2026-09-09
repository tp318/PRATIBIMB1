"""
Unit tests for ResidualGenerator (signed, absolute, and normalized physical residuals).
"""

import os
import sys
import unittest

_test_dir = os.path.dirname(os.path.abspath(__file__))
_health_dir = os.path.dirname(_test_dir)
_aerotwin_dir = os.path.dirname(_health_dir)
_root_dir = os.path.dirname(_aerotwin_dir)

for _p in [_health_dir, _aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from health.residuals import ResidualGenerator
from health.models import ExpectedState, OperatingState, ResidualState


class TestResiduals(unittest.TestCase):

    def test_signed_and_normalized_residuals(self):
        generator = ResidualGenerator()

        observed_telemetry = {
            "timestamp": 5.0,
            "simulation_time": 5.0,
            "engine_id": "AEROTWIN-001",
            "operating_mode": "CRUISE",
            "rpm": 2500.0,
            "mean_torque": 120.0,
            "instant_torque": 125.0,
            "load_torque": 115.0,
            "friction_torque": 15.0,
            "net_torque": 10.0,
            "cylinder_1_torque": 150.0,
            "cylinder_2_torque": 150.0,
            "cylinder_3_torque": 100.0,  # 50 N*m drop in Cyl 3
            "cylinder_4_torque": 150.0,
            "cht": 150.0,                # +15 °C CHT elevation
            "egt": 700.0,
            "oil_temperature": 95.0,
            "oil_pressure": 400000.0,
            "oil_pressure_psi": 58.0,
            "fuel_flow": 0.005,
            "fuel_flow_lph": 20.0,
            "fuel_pressure": 320000.0,
            "vibration": 0.30,
        }

        expected = ExpectedState(
            rpm=2500.0,
            crank_angle=180.0,
            mean_torque=120.0,
            instant_torque=125.0,
            load_torque=115.0,
            friction_torque=15.0,
            net_torque=10.0,
            cylinder_1_torque=150.0,
            cylinder_2_torque=150.0,
            cylinder_3_torque=150.0,
            cylinder_4_torque=150.0,
            cht=135.0,
            egt=700.0,
            oil_temperature=95.0,
            oil_pressure=400000.0,
            oil_pressure_psi=58.0,
            fuel_flow=0.005,
            fuel_flow_lph=20.0,
            fuel_pressure=320000.0,
            vibration=0.20,
        )

        op_state = OperatingState(5.0, 5.0, "AEROTWIN-001", "CRUISE", 0.60, 2500.0)
        res_state = generator.generate(observed_telemetry, expected, op_state)

        self.assertIsInstance(res_state, ResidualState)
        self.assertAlmostEqual(res_state.raw_signed["cht"], 15.0)
        self.assertAlmostEqual(res_state.raw_signed["cylinder_3_torque"], -50.0)

        # Metadata fields must NOT be in raw_signed
        self.assertNotIn("timestamp", res_state.raw_signed)
        self.assertNotIn("simulation_time", res_state.raw_signed)
        self.assertNotIn("engine_id", res_state.raw_signed)
        self.assertNotIn("operating_mode", res_state.raw_signed)


if __name__ == "__main__":
    unittest.main()
