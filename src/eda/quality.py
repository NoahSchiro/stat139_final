"""EDA on data quality and track coverage."""

from collections.abc import Iterator
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

from src.eda import pct, save_fig


def data_quality_checks(df: pd.DataFrame) -> None:
    """What data-quality issues remain in the cleaned table (missing values, duration outliers)?"""
    missing_rows = df.isna().sum()
    missing_rows = missing_rows[missing_rows > 0].sort_values(ascending=False)
    print("  missing values per play-level column:")
    print(missing_rows.map(lambda n: f"{int(n):,} ({pct(n / len(df))})").to_string())

    tracks = df.drop_duplicates("track_id_clean")
    missing_tracks = tracks[["tempo", "loudness", "time_signature"]].isna().sum()
    print(
        "  missing values per unique track:",
        {c: int(v) for c, v in missing_tracks.items()},
    )
    print(
        f"  plays with no outcome flag set (unlabeled skip_class): {int((df['skip_class'] == 'unknown').sum()):,}"
    )

    duration = df["duration"]
    low, high = duration.quantile([0.01, 0.99])
    print(
        f"  duration percentiles: 1% {low:.0f}s, median {duration.median():.0f}s, 99% {high:.0f}s, max {duration.max():.0f}s"
    )
    print(f"  plays longer than 10 minutes: {int((duration > 600).sum()):,}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(missing_rows.index, missing_rows.values)
    axes[0].set_ylabel("plays with missing value")
    axes[0].set_title("Missing values in cleaned table")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].hist(duration, bins=60)
    axes[1].axvline(low, color="gray", ls="--", label="1st percentile")
    axes[1].axvline(high, color="gray", ls="--", label="99th percentile")
    axes[1].set_xlabel("duration (s)")
    axes[1].set_ylabel("plays")
    axes[1].set_title("Duration distribution with outlier thresholds")
    axes[1].legend()
    save_fig(fig, "21_data_quality")


def track_reuse(df: pd.DataFrame) -> None:
    """How are tracks reused across sessions, and does reuse frequency relate to skipping?"""
    per_track = df.groupby("track_id_clean")["session_id"].nunique()
    reuse = df["track_id_clean"].map(per_track)
    dist = per_track.value_counts().sort_index()
    quartile = pd.qcut(reuse, q=4, duplicates="drop")
    rate = df.groupby(quartile, observed=True)["skip_2"].mean()

    print(f"  unique tracks in the log: {len(per_track):,}")
    print(
        f"  average sessions per track: {per_track.mean():.2f}; tracks in only one session: {pct((per_track == 1).mean())}"
    )
    print(f"  max sessions containing one track: {int(per_track.max())}")
    print("  skip rate by per-track reuse quartile:")
    for interval, r in cast("Iterator[tuple[pd.Interval, float]]", rate.items()):
        print(f"    {int(interval.left)}-{int(interval.right)} sessions: {pct(r)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(dist.index, dist.values)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("sessions containing the track")
    axes[0].set_ylabel("tracks (log)")
    axes[0].set_title("Track reuse across sessions")
    labels = [f"{int(i.left)}-{int(i.right)}" for i in rate.index]
    axes[1].bar(range(len(rate)), rate.values)
    axes[1].set_xticks(range(len(rate)), labels)
    axes[1].set_xlabel("sessions containing the track")
    axes[1].set_ylabel("skip_2 rate")
    axes[1].set_title("Skip rate by track reuse quartile")
    save_fig(fig, "22_track_coverage")
