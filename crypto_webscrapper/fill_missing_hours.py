"""Fill missing hourly records in bitcoin_news_final_dataset.csv.

- Keeps existing news rows unchanged
- For each day that has at least one news row, ensures every hour (00..23) has at least one row
- Inserts a synthetic row for hours without news: only date-related columns are set; others are empty (NaN)
- Writes an intermediate Feather file for the hourly-complete dataset
- Overwrites the CSV with the hourly-complete dataset (as requested)
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

DATA_DIR = Path(__file__).parent
CSV_PATH = DATA_DIR / "bitcoin_news_final_dataset.csv"
FEATHER_PATH = DATA_DIR / "bitcoin_news_final_dataset_hourly.feather"


def _infer_date_columns(df: pd.DataFrame) -> List[str]:
    """Heuristically determine which columns are date-related to set on synthetic rows.

    Rules:
    - Always include 'date' if present (primary timestamp)
    - Include columns with names containing 'date' (case-insensitive)
    - Include simple calendar fields like 'year', 'month', 'day', 'hour' if present
    - Exclude columns that are clearly price/metrics/text
    """
    date_cols: List[str] = []
    lc = [c.lower() for c in df.columns]
    for col, lcol in zip(df.columns, lc):
        if col == "date":
            date_cols.append(col)
        elif "date" in lcol:
            date_cols.append(col)
        elif lcol in {"year", "month", "day", "hour"}:
            date_cols.append(col)
    # De-duplicate while preserving order
    seen = set()
    ordered = []
    for c in date_cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def fill_missing_hours(csv_path: Path = CSV_PATH, feather_path: Path = FEATHER_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Load
    df = pd.read_csv(csv_path, low_memory=False)
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in the dataset")

    # Parse date; do not alter original 'date' values for existing rows
    date_parsed = pd.to_datetime(df["date"], errors="coerce")
    if date_parsed.isna().all():
        raise ValueError("All 'date' values failed to parse; cannot proceed")

    # Determine existing hourly buckets
    date_hour = date_parsed.dt.floor("H")

    # Only operate on days that are present in the dataset
    days_present = pd.to_datetime(date_parsed.dt.date)
    unique_days = pd.DatetimeIndex(sorted(days_present.dropna().unique()))

    # Build set of existing hours
    existing_hours = set(date_hour.dropna().unique())

    # Identify which columns should be set on synthetic rows
    date_columns = _infer_date_columns(df)

    # Prepare synthetic rows
    synth_rows = []

    # For each day, generate all 24 hours and add a synthetic row if no record exists in that hour
    for day in unique_days:
        # generate 24 timestamps for this day at the top of each hour
        start = pd.Timestamp(day.normalize())
        hours = pd.date_range(start=start, periods=24, freq="H")
        for ts in hours:
            if ts not in existing_hours:
                # Construct a dict with NaNs for all columns
                row = {col: pd.NA for col in df.columns}

                # Set date-related fields
                if "date" in date_columns:
                    row["date"] = ts
                for col in date_columns:
                    if col == "date":
                        continue
                    lcol = col.lower()
                    try:
                        if lcol == "year":
                            row[col] = ts.year
                        elif lcol == "month":
                            row[col] = ts.month
                        elif lcol == "day":
                            row[col] = ts.day
                        elif lcol == "hour":
                            row[col] = ts.hour
                        else:
                            # If it's some other *date* column (e.g., 'begins_at', 'published_date'), set to ts
                            row[col] = ts
                    except Exception:
                        # Fallback to NA if assignment incompatible
                        row[col] = pd.NA

                synth_rows.append(row)

    if synth_rows:
        synth_df = pd.DataFrame(synth_rows, columns=df.columns)
        # Concatenate and sort by date
        # Use a stable sort to keep original rows' relative order
        combined = pd.concat([df, synth_df], ignore_index=True)
    else:
        combined = df.copy()

    # Ensure 'date' column remains as datetime for saving feather, but store CSV as string
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")

    # Sort by date ascending to make it easier to analyze by time
    combined = combined.sort_values("date").reset_index(drop=True)

    # Save intermediate Feather (requires pyarrow)
    try:
        combined.to_feather(feather_path)
    except Exception as e:
        # If pyarrow is missing or another error occurs, at least inform the user in stdout
        print(f"Warning: failed to write Feather file at {feather_path}: {e}")

    # Save back to CSV (as requested), formatting date like original
    # Convert date to string in ISO format with seconds
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    combined.to_csv(csv_path, index=False, encoding="utf-8")

    return combined


def main():
    out_df = fill_missing_hours()
    total_days = out_df["date"].str[:10].nunique()
    print(f"Hourly-complete dataset saved. Rows: {len(out_df):,}, Days: {total_days}")
    if FEATHER_PATH.exists():
        print(f"Intermediate Feather saved to: {FEATHER_PATH}")
    print(f"CSV overwritten at: {CSV_PATH}")


if __name__ == "__main__":
    main()
