"""
P1 - Data Management (UC1)
DatasetLoader: FR-DM-01 - load the CICIDS 2017 dataset from native CSV.

Concatenates all CSVs found in a directory, handling the well-known
CICIDS 2017 quirks:
  * leading/trailing whitespace in column names,
  * mixed encodings (one file ships as latin-1),
  * a duplicated 'Fwd Header Length' column in some releases.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads and concatenates the raw CICIDS 2017 flow CSVs (FR-DM-01)."""

    def __init__(self, csv_directory: str):
        self.csv_directory = Path(csv_directory)
        self.raw_dataframe: pd.DataFrame | None = None
        self.expected_columns: list[str] = []

    # ------------------------------------------------------------------ API
    def load_all_csvs(self) -> pd.DataFrame:
        """Read every *.csv in ``csv_directory`` into one DataFrame."""
        csv_paths = sorted(self.csv_directory.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(
                f"No CSV files found in {self.csv_directory}. "
                "Place the CICIDS 2017 flow CSVs there first."
            )
        frames = [self._read_single_csv(p) for p in csv_paths]
        self.raw_dataframe = pd.concat(frames, ignore_index=True)
        logger.info(
            "Loaded %d CSV files, %d rows, %d columns",
            len(csv_paths), len(self.raw_dataframe), self.raw_dataframe.shape[1],
        )
        return self.raw_dataframe

    def validate_schema(self, expected: list[str]) -> bool:
        """Check that all expected columns are present (activity-diagram
        'Schema valid?' gate). Logs missing columns and returns False
        instead of raising, so the caller decides how to fail."""
        if self.raw_dataframe is None:
            raise RuntimeError("Call load_all_csvs() before validate_schema().")
        self.expected_columns = list(expected)
        missing = [c for c in expected if c not in self.raw_dataframe.columns]
        if missing:
            logger.error("Schema validation failed; missing columns: %s", missing)
            return False
        return True

    def get_row_count(self) -> int:
        if self.raw_dataframe is None:
            return 0
        return len(self.raw_dataframe)

    # -------------------------------------------------------------- helpers
    def _read_single_csv(self, path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            # Thursday-WorkingHours ships as latin-1 in some releases.
            df = pd.read_csv(path, low_memory=False, encoding="latin-1")
        df = self._strip_whitespace_columns(df)
        # Drop the duplicated 'Fwd Header Length' column if present.
        df = df.loc[:, ~df.columns.duplicated()]
        logger.debug("Read %s: %d rows", path.name, len(df))
        return df

    @staticmethod
    def _strip_whitespace_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        return df
