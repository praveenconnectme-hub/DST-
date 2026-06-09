"""
Module 3 — Signal ingestion (BRD §4.4, Phase 2).

Loads three correlated exogenous signals into the signal_data table:
  - temp_deviation          (from weather_data.csv)
  - competitor_price_index  (from competitor_scrapes.csv)
  - search_trend_index      (from google_trends_export.csv)

All file I/O goes through repo.read_csv_raw() — Rule 1 compliant.
D-016: weather_data.csv is written by synthetic.py alongside the JSON
so that all three sources are loadable via a single read_csv_raw() path.
"""
import os

# (filename, value_column_in_file, signal_name_in_db, source_connector)
_SIGNAL_FILES = [
    ("weather_data.csv",         "temp_deviation",         "temp_deviation",         "synthetic_weather"),
    ("competitor_scrapes.csv",   "competitor_price_index", "competitor_price_index", "synthetic_competitor"),
    ("google_trends_export.csv", "search_trend_index",     "search_trend_index",     "synthetic_trends"),
]


def run(repo, data_dir: str) -> dict:
    """
    Read signal CSV files and persist rows to signal_data via repository.

    Parameters
    ----------
    repo     : AbstractRepository
    data_dir : directory containing the three signal CSV files

    Returns
    -------
    dict with keys 'signals_loaded' (int) and 'total_rows' (int)
    """
    total_rows = 0

    for fname, value_col, signal_name, source_connector in _SIGNAL_FILES:
        path = os.path.join(data_dir, fname)
        df = repo.read_csv_raw(path)

        rows = [
            {
                "signal_name":      signal_name,
                "state_code":       r["state_code"],
                "week_index":       r["week_index"],
                "value":            float(r[value_col]),
                "source_connector": source_connector,
            }
            for _, r in df.iterrows()
        ]

        repo.upsert("signal_data", rows)
        total_rows += len(rows)

    return {
        "signals_loaded": len(_SIGNAL_FILES),
        "total_rows":     total_rows,
    }
