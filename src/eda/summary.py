"""Descriptive summary of every variable: missingness, central tendency, spread, and levels."""

import pandas as pd

from src.eda import pct

NUMERIC_LEVEL_COLUMNS = ["key", "mode", "time_signature"]


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize each numeric variable as missing/non-missing counts, mean, median, sd, and IQR."""
    numeric = df.select_dtypes("number").drop(
        columns=NUMERIC_LEVEL_COLUMNS, errors="ignore"
    )
    return pd.DataFrame(
        {
            "non_missing": numeric.notna().sum(),
            "missing": numeric.isna().sum(),
            "mean": numeric.mean(),
            "median": numeric.median(),
            "sd": numeric.std(),
            "iqr": numeric.quantile(0.75) - numeric.quantile(0.25),
        }
    )


def categorical_names(df: pd.DataFrame) -> list[str]:
    """List categorical variables: string/categorical/bool columns (minus identifiers) plus numerically-encoded levels."""
    categorical = df.select_dtypes(include=["object", "str", "category", "bool"]).drop(
        columns=["session_id", "track_id_clean"], errors="ignore"
    )
    present_levels = [name for name in NUMERIC_LEVEL_COLUMNS if name in df.columns]
    return [*categorical.columns, *present_levels]


def descriptive_summary(df: pd.DataFrame) -> None:
    """What does each variable look like: missingness, central tendency, spread, and level counts?"""
    stats = numeric_summary(df)
    print(
        f"  numeric variables ({len(stats)}): non-missing/missing, mean, median, sd, IQR"
    )
    print(stats.round(3).to_string())

    names = categorical_names(df)
    print(
        f"  categorical variables ({len(names)}): non-missing/missing, levels, counts per level"
    )
    for name in names:
        col = df[name]
        counts = col.value_counts().sort_index()
        missing = int(col.isna().sum())
        print(
            f"  {name}: {len(counts)} levels, "
            f"{len(col) - missing:,} non-missing, {missing:,} missing"
        )
        for level, count in counts.items():
            print(f"    {level}: {int(count):,} ({pct(count / len(col))})")
