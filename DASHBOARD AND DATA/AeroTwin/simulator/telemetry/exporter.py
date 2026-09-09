"""
AeroTwin-4 Telemetry Exporter.

Utility for exporting buffered EngineTelemetry payloads to CSV, Pandas DataFrames,
and Parquet files.
"""

import os
import warnings
from typing import List, Union
import pandas as pd

from .schema import EngineTelemetry


class TelemetryExporter:
    """
    Exports EngineTelemetry lists to CSV and Parquet formats for data processing pipelines.
    """

    def __init__(self, telemetry_data: List[EngineTelemetry] = None):
        self.data = telemetry_data or []

    def add(self, telemetry: EngineTelemetry):
        """
        Add a telemetry record.
        """
        self.data.append(telemetry)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert telemetry history to Pandas DataFrame.
        """
        dict_list = [t.to_dict() for t in self.data]
        return pd.DataFrame(dict_list)

    def to_csv(self, file_path: str) -> str:
        """
        Export telemetry history to CSV file.
        """
        df = self.to_dataframe()
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        df.to_csv(file_path, index=False)
        return os.path.abspath(file_path)

    def to_parquet(self, file_path: str) -> str:
        """
        Export telemetry history to Parquet file.
        """
        df = self.to_dataframe()
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        try:
            df.to_parquet(file_path, index=False)
        except ImportError:
            # No parquet engine installed. Fall back to CSV, but say so - silently
            # returning a .csv path from to_parquet() is how the caller ended up
            # reporting "Exported Parquet" for a file that was never written.
            alt_path = file_path.replace(".parquet", ".csv")
            warnings.warn(
                "No parquet engine (pyarrow/fastparquet) available; "
                f"wrote CSV to {alt_path} instead.",
                RuntimeWarning,
                stacklevel=2,
            )
            df.to_csv(alt_path, index=False)
            return os.path.abspath(alt_path)
        return os.path.abspath(file_path)
