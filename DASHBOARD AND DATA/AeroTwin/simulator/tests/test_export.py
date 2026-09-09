"""
Unit tests for AeroTwin-4 Telemetry Export & Pandas integration.
"""

import os
import sys
import tempfile
import unittest

_test_dir = os.path.dirname(os.path.abspath(__file__))
_sim_dir = os.path.dirname(_test_dir)
_aerotwin_dir = os.path.dirname(_sim_dir)
_root_dir = os.path.dirname(_aerotwin_dir)

for _p in [_sim_dir, _aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from simulator.runner import EngineRunner
    from simulator.telemetry.exporter import TelemetryExporter
except ImportError:
    from runner import EngineRunner
    from telemetry.exporter import TelemetryExporter


class TestTelemetryExport(unittest.TestCase):

    def test_dataframe_and_csv_export(self):
        runner = EngineRunner(dt=0.01)
        history = runner.run_for(2.0)

        exporter = TelemetryExporter(history)
        df = exporter.to_dataframe()

        self.assertEqual(len(df), len(history))
        self.assertIn("rpm", df.columns)
        self.assertIn("cht", df.columns)
        self.assertIn("oil_pressure", df.columns)
        self.assertIn("vibration", df.columns)

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "test_telemetry.csv")
            exported_file = exporter.to_csv(csv_path)

            self.assertTrue(os.path.exists(exported_file))
            self.assertGreater(os.path.getsize(exported_file), 0)


if __name__ == "__main__":
    unittest.main()
