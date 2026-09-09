"""
Unit tests for Counterfactual Digital Twin state estimation & degradation residuals.
"""

import os
import sys
import unittest
import numpy as np

_test_dir = os.path.dirname(os.path.abspath(__file__))
_health_dir = os.path.dirname(_test_dir)
_aerotwin_dir = os.path.dirname(_health_dir)
_root_dir = os.path.dirname(_aerotwin_dir)

for _p in [_health_dir, _aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simulator.runner import EngineRunner
from degradation.config import DegradationConfig, DegradationType, ComponentID
from degradation.injector import DegradationInjector
from health.engine import DigitalTwinStateEngine


class TestDegradationResiduals(unittest.TestCase):

    def test_healthy_vs_healthy_residual_near_zero(self):
        # Run healthy simulation
        h_runner = EngineRunner(dt=0.01, seed=42)
        h_inj = DegradationInjector(config=DegradationConfig.healthy(), runner=h_runner, run_id="RUN_H")
        telemetry_list, _ = h_inj.run_simulation(duration_seconds=3.0)

        # Process with counterfactual Healthy DigitalTwinStateEngine (same seed=42)
        dt_engine = DigitalTwinStateEngine(dt=0.01, seed=42, mode="COUNTERFACTUAL")
        dt_frames = [dt_engine.process_telemetry(t) for t in telemetry_list]

        # In healthy-vs-healthy comparison, residuals should be near zero
        rpm_res = np.mean([abs(f.residuals.raw_signed["rpm"]) for f in dt_frames[50:]])
        cht_res = np.mean([abs(f.residuals.raw_signed["cht"]) for f in dt_frames[50:]])
        self.assertLess(rpm_res, 1.0)
        self.assertLess(cht_res, 0.5)

    def test_cylinder_3_degradation_residual_isolation(self):
        # Run Cylinder 3 degradation (severity = 0.50)
        config = DegradationConfig.single_fault(DegradationType.CYLINDER, ComponentID.CYLINDER_3, 0.50)
        deg_runner = EngineRunner(dt=0.01, seed=42)
        deg_inj = DegradationInjector(config=config, runner=deg_runner, run_id="RUN_CYL3")
        telemetry_list, _ = deg_inj.run_simulation(duration_seconds=3.0)

        dt_engine = DigitalTwinStateEngine(dt=0.01, seed=42, mode="COUNTERFACTUAL")
        dt_frames = [dt_engine.process_telemetry(t) for t in telemetry_list]

        # Verify Cylinder 3 torque residual is large negative while Cylinders 1, 2, 4 remain small
        c3_res_max = np.max([abs(f.residuals.raw_signed["cylinder_3_torque"]) for f in dt_frames[50:]])
        c1_res_max = np.max([abs(f.residuals.raw_signed["cylinder_1_torque"]) for f in dt_frames[50:]])

        self.assertGreater(c3_res_max, 5.0)
        self.assertLess(c1_res_max, c3_res_max * 0.40)

    def test_bearing_degradation_friction_residual(self):
        config = DegradationConfig.single_fault(DegradationType.BEARING, ComponentID.BEARING, 0.50)
        b_runner = EngineRunner(dt=0.01, seed=42)
        b_inj = DegradationInjector(config=config, runner=b_runner, run_id="RUN_BEARING")
        telemetry_list, _ = b_inj.run_simulation(duration_seconds=3.0)

        dt_engine = DigitalTwinStateEngine(dt=0.01, seed=42, mode="COUNTERFACTUAL")
        dt_frames = [dt_engine.process_telemetry(t) for t in telemetry_list]

        fric_res_mean = np.mean([f.residuals.raw_signed["friction_torque"] for f in dt_frames[50:]])
        self.assertGreater(fric_res_mean, 0.5)

    def test_cooling_degradation_thermal_residual(self):
        config = DegradationConfig.single_fault(DegradationType.COOLING, ComponentID.COOLING_SYSTEM, 0.50)
        c_runner = EngineRunner(dt=0.01, seed=42)
        c_inj = DegradationInjector(config=config, runner=c_runner, run_id="RUN_COOLING")
        telemetry_list, _ = c_inj.run_simulation(duration_seconds=10.0)

        dt_engine = DigitalTwinStateEngine(dt=0.01, seed=42, mode="COUNTERFACTUAL")
        dt_frames = [dt_engine.process_telemetry(t) for t in telemetry_list]

        cht_res_final = dt_frames[-1].residuals.raw_signed["cht"]
        self.assertGreater(cht_res_final, 3.0)

    def test_lubrication_degradation_pressure_residual(self):
        config = DegradationConfig.single_fault(DegradationType.LUBRICATION, ComponentID.LUBRICATION_SYSTEM, 0.50)
        l_runner = EngineRunner(dt=0.01, seed=42)
        l_inj = DegradationInjector(config=config, runner=l_runner, run_id="RUN_LUB")
        telemetry_list, _ = l_inj.run_simulation(duration_seconds=3.0)

        dt_engine = DigitalTwinStateEngine(dt=0.01, seed=42, mode="COUNTERFACTUAL")
        dt_frames = [dt_engine.process_telemetry(t) for t in telemetry_list]

        oil_p_res_mean = np.mean([f.residuals.raw_signed["oil_pressure"] for f in dt_frames[50:]])
        self.assertLess(oil_p_res_mean, -20.0)


if __name__ == "__main__":
    unittest.main()
