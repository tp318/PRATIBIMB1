"""
============================================================================
flight_replay.py — Sortie Replay Engine from Dataset
============================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import pandas as pd

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / "DATASET" / "pratibimb_synthetic_residual_dataset.csv"

FAULT_NAMES: Dict[int, str] = {
    0: "Normal",
    1: "Misfire",
    2: "Injector",
    3: "Cooling",
    4: "Lubrication",
    5: "Sensor",
    6: "Combustion",
    7: "Overheating",
    8: "Vibration",
}


@dataclass
class FlightSample:
    engine_id: int
    step: int
    time_sec: float
    fault_label: int
    fault_name: str
    rul_hours: float
    residuals: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "step": self.step,
            "time_sec": round(self.time_sec, 2),
            "fault_label": self.fault_label,
            "fault_name": self.fault_name,
            "rul_hours": round(self.rul_hours, 5),
            "residuals": self.residuals,
        }


class FlightReplayer:
    """
    Flight Replay Engine for PRATIBIMB Digital Twin.
    Streams real flight residual trajectories recorded at 50 Hz.
    """

    def __init__(
        self,
        dataset_path: Optional[Path | str] = None,
        fallback_to_synthetic: bool = True,
    ) -> None:
        self.dataset_path = Path(dataset_path or DEFAULT_DATASET_PATH)
        self.fallback_to_synthetic = fallback_to_synthetic
        self._df_cache: Optional[pd.DataFrame] = None
        self._engine_cache: Dict[int, pd.DataFrame] = {}

    def _generate_synthetic_trajectory(self, engine_id: int, num_steps: int = 1500) -> pd.DataFrame:
        """Generates a physically realistic synthetic trajectory when CSV is absent."""
        np.random.seed(engine_id * 42)
        # Determine simulated fault for this engine
        fault_label = (engine_id % 9)
        t = np.arange(num_steps) * 0.02

        # Base nominal residuals with small noise
        rpm_res = np.random.normal(0.0, 15.0, num_steps)
        cht_res = np.random.normal(0.0, 1.2, num_steps)
        egt_res = np.random.normal(0.0, 4.0, num_steps)
        oil_p_res = np.random.normal(0.0, 0.8, num_steps)
        oil_t_res = np.random.normal(0.0, 0.9, num_steps)
        fuel_res = np.random.normal(0.0, 0.4, num_steps)
        vib_res = np.random.normal(0.0, 0.15, num_steps)
        volt_res = np.random.normal(0.0, 0.05, num_steps)
        inj_res = np.random.normal(0.0, 0.2, num_steps)

        # Inject characteristic degradation signature after step 300
        fault_start = 300
        if fault_label == 1:  # Misfire
            rpm_res[fault_start:] -= np.linspace(50.0, 280.0, num_steps - fault_start)
            vib_res[fault_start:] += np.linspace(0.8, 3.5, num_steps - fault_start)
        elif fault_label == 3:  # Cooling
            cht_res[fault_start:] += np.linspace(10.0, 48.0, num_steps - fault_start)
        elif fault_label == 4:  # Lubrication
            oil_p_res[fault_start:] -= np.linspace(5.0, 22.0, num_steps - fault_start)
            oil_t_res[fault_start:] += np.linspace(4.0, 25.0, num_steps - fault_start)
        elif fault_label == 7:  # Overheating
            cht_res[fault_start:] += np.linspace(15.0, 60.0, num_steps - fault_start)
            egt_res[fault_start:] += np.linspace(30.0, 110.0, num_steps - fault_start)

        # RUL decay profile
        max_rul = 0.25  # 15 minutes in hours
        rul_hours = np.maximum(0.01, max_rul - (t / (num_steps * 0.02)) * (max_rul - 0.02))

        return pd.DataFrame({
            "engine_id": engine_id,
            "fault_label": fault_label,
            "rul_hours": rul_hours,
            "rpm_residual": rpm_res,
            "cht_residual": cht_res,
            "egt_residual": egt_res,
            "oil_pressure_residual": oil_p_res,
            "oil_temp_residual": oil_t_res,
            "fuel_flow_residual": fuel_res,
            "vibration_residual": vib_res,
            "batt_voltage_residual": volt_res,
            "inj_timing_residual": inj_res,
        })

    def load_engine_data(self, engine_id: int) -> pd.DataFrame:
        """Loads all timesteps for a specific engine trajectory."""
        if engine_id in self._engine_cache:
            return self._engine_cache[engine_id]

        if not self.dataset_path.exists():
            if self.fallback_to_synthetic:
                synth_df = self._generate_synthetic_trajectory(engine_id)
                self._engine_cache[engine_id] = synth_df
                return synth_df
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        # If cache not yet loaded, filter dataset
        cols = [
            "engine_id", "fault_label", "rul_hours",
            "rpm_residual", "cht_residual", "egt_residual",
            "oil_pressure_residual", "oil_temp_residual", "fuel_flow_residual",
            "vibration_residual", "batt_voltage_residual", "inj_timing_residual",
        ]
        # Read engine rows efficiently
        chunks = []
        for chunk in pd.read_csv(self.dataset_path, usecols=cols, chunksize=100000):
            eng_chunk = chunk[chunk["engine_id"] == engine_id]
            if len(eng_chunk) > 0:
                chunks.append(eng_chunk)
        
        if not chunks:
            if self.fallback_to_synthetic:
                synth_df = self._generate_synthetic_trajectory(engine_id)
                self._engine_cache[engine_id] = synth_df
                return synth_df
            raise ValueError(f"Engine ID {engine_id} not found in dataset.")

        eng_df = pd.concat(chunks, ignore_index=True)
        self._engine_cache[engine_id] = eng_df
        return eng_df

    def stream_engine(
        self,
        engine_id: int,
        start_step: int = 0,
        max_steps: Optional[int] = None,
        playback_rate_hz: float = 0.0,  # 0.0 means unthrottled generator
    ) -> Generator[FlightSample, None, None]:
        """
        Yields flight samples sequentially at 50 Hz.
        """
        df = self.load_engine_data(engine_id)
        total_steps = len(df)
        end_step = min(total_steps, start_step + (max_steps or total_steps))

        delay_s = 1.0 / playback_rate_hz if playback_rate_hz > 0.0 else 0.0

        for i in range(start_step, end_step):
            row = df.iloc[i]
            sample = FlightSample(
                engine_id=int(row["engine_id"]),
                step=i,
                time_sec=i * 0.02,  # 50 Hz sample interval
                fault_label=int(row["fault_label"]),
                fault_name=FAULT_NAMES.get(int(row["fault_label"]), "Unknown"),
                rul_hours=float(row["rul_hours"]),
                residuals={
                    "rpm_residual": float(row["rpm_residual"]),
                    "cht_residual": float(row["cht_residual"]),
                    "egt_residual": float(row["egt_residual"]),
                    "oil_pressure_residual": float(row["oil_pressure_residual"]),
                    "oil_temp_residual": float(row["oil_temp_residual"]),
                    "fuel_flow_residual": float(row["fuel_flow_residual"]),
                    "vibration_residual": float(row["vibration_residual"]),
                    "batt_voltage_residual": float(row["batt_voltage_residual"]),
                    "inj_timing_residual": float(row["inj_timing_residual"]),
                },
            )
            if delay_s > 0.0:
                time.sleep(delay_s)
            yield sample

    # Convenience alias for test harnesses
    stream_engine_samples = stream_engine
