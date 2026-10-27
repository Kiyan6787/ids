"""Loads the CICIDS 2017 dataset from its native CSV files.
"""

import logging
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads and concatenates the raw CICIDS 2017 flow CSVs."""

    def __init__(self, csv_directory: str):
        self.csv_directory = Path(csv_directory)
        self.raw_dataframe = None

    def load_all_csvs(self) -> pd.DataFrame:
        rank = {name: i for i, name in enumerate(config.CSV_FILE_ORDER)}
        csv_paths = sorted(self.csv_directory.glob("*.csv"),
                           key=lambda p: (rank.get(p.name, len(rank)), p.name))
        unknown = [p.name for p in csv_paths if p.name not in rank]
        if unknown:
            logger.warning("Not in config.CSV_FILE_ORDER, appended at the end: %s", unknown)
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
        """Check that all expected columns are present. Logs missing
        columns and returns False instead of raising, so the caller
        decides how to fail."""
        if self.raw_dataframe is None:
            raise RuntimeError("Call load_all_csvs() before validate_schema().")
        missing = [c for c in expected if c not in self.raw_dataframe.columns]
        if missing:
            logger.error("Schema validation failed; missing columns: %s", missing)
            return False
        return True

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _read_single_csv(path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, low_memory=False, encoding="latin-1")
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.duplicated()]
        logger.debug("Read %s: %d rows", path.name, len(df))
        return df
