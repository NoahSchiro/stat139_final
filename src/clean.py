"""Load the raw Spotify session data, clean it, and write the merged analysis table.

Cleaning policy (value-level fixes; no rows are dropped):
- parse `date` to datetime and force behavioral columns to bool
- mask impossible track-feature values as missing: `tempo` == 0, `loudness` <= -60,
  `time_signature` == 0 (a real recording cannot have these values)
- derive `skip_class`, an ordered category holding the most severe skip flag
  (skip_1 = briefest listen < skip_2 < skip_3 < completed); plays with no skip flag
  set at all (~2%) have no outcome label and are classified as 'unknown'

Paths are relative to the project root; run via `python main.py`.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def to_bool(col: pd.Series) -> pd.Series:
    """Coerce a column to bool, accepting bools, 0/1 ints, or 'true'/'false' strings."""
    if col.dtype == bool:
        return col
    if pd.api.types.is_integer_dtype(col):
        return col.eq(1)
    mapped = col.str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise ValueError(
            f"column {col.name!r} contains values that are neither bool nor 'true'/'false'"
        )
    return mapped


def load_raw_log(path: Path = Path("data/log_mini.csv")) -> pd.DataFrame:
    """Read the session log with dates parsed and boolean columns normalized."""
    log = pd.read_csv(path, parse_dates=["date"])
    bool_cols = [
        "skip_1",
        "skip_2",
        "skip_3",
        "not_skipped",
        "context_switch",
        "no_pause_before_play",
        "short_pause_before_play",
        "long_pause_before_play",
        "hist_user_behavior_is_shuffle",
        "premium",
    ]
    log[bool_cols] = log[bool_cols].apply(to_bool)
    return log


def load_raw_track_features(path: Path = Path("data/tf_mini.csv")) -> pd.DataFrame:
    """Read the raw track-feature table."""
    return pd.read_csv(path)


def clean_track_features(tracks: pd.DataFrame) -> pd.DataFrame:
    """Mask impossible track-feature values as missing; does not modify its input."""
    cleaned = tracks.copy()
    cleaned.loc[cleaned["tempo"].eq(0), "tempo"] = np.nan
    cleaned.loc[cleaned["loudness"] <= -60, "loudness"] = np.nan
    cleaned.loc[cleaned["time_signature"].eq(0), "time_signature"] = pd.NA
    cleaned["time_signature"] = cleaned["time_signature"].astype("Int64")
    return cleaned


def derive_skip_class(df: pd.DataFrame) -> pd.Categorical[str]:
    """Classify each play by its most severe true skip flag ('unknown' if no flag is set)."""
    classes = pd.Series("unknown", index=df.index, dtype="object")
    classes[df["not_skipped"]] = "completed"
    classes[df["skip_3"]] = "skip_3"
    classes[df["skip_2"]] = "skip_2"
    classes[df["skip_1"]] = "skip_1"
    return pd.Categorical(
        classes,
        categories=["completed", "skip_3", "skip_2", "skip_1", "unknown"],
        ordered=True,
    )


def build_analysis_table(log: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """Join plays to track features and derive the skip_class outcome column.

    Raises ValueError if any play has no matching track features.
    """
    df = log.merge(
        tracks,
        left_on="track_id_clean",
        right_on="track_id",
        how="left",
        validate="m:1",
    )
    unmatched = int(df["track_id"].isna().sum())
    if unmatched:
        raise ValueError(
            f"{unmatched} plays reference tracks missing from the feature table"
        )
    df = df.drop(columns=["track_id"])
    df["skip_class"] = derive_skip_class(df)
    return df


def validate_log(log: pd.DataFrame) -> None:
    """Raise ValueError if the session log violates its structural invariants."""
    dupes = int(log.duplicated(["session_id", "session_position"]).sum())
    if dupes:
        raise ValueError(f"{dupes} duplicate (session_id, session_position) rows")
    outside = log["session_position"].gt(log["session_length"]) | log[
        "session_position"
    ].lt(1)
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} rows have session_position outside 1..session_length"
        )


def clean(output_path: Path) -> Path:
    """Load, clean, and merge the raw data; write the analysis table and return its path."""
    log = load_raw_log()
    validate_log(log)
    tracks = clean_track_features(load_raw_track_features())
    df = build_analysis_table(log, tracks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    masked = {
        c: int(tracks[c].isna().sum()) for c in ("tempo", "loudness", "time_signature")
    }
    print(
        f"clean: wrote {len(df):,} plays ({log['session_id'].nunique():,} sessions, "
        f"{tracks['track_id'].nunique():,} tracks) to {output_path}; "
        f"tracks with masked invalid values: {masked}"
    )
    return output_path
