"""Collapse the play-level table to the two analysis units from the proposal.

- track level (one row per track, `MIN_PLAYS` or more plays): skip rates, the
  play count used as a WLS weight, the track's audio features, and context-mix
  covariates aggregated from its plays
- session x context level (one row per session and context type): the skip rate
  that is one observation in the ANOVA across listening contexts
"""

from pathlib import Path

import pandas as pd

MIN_PLAYS = 5

AUDIO_FEATURES = (
    "acousticness",
    "beat_strength",
    "bounciness",
    "danceability",
    "dyn_range_mean",
    "energy",
    "flatness",
    "instrumentalness",
    "liveness",
    "loudness",
    "mechanism",
    "organism",
    "speechiness",
    "tempo",
    "valence",
    "duration",
)
TRACK_CONTROLS = ("us_popularity_estimate", "release_year")
TRACK_CATEGORICAL = ("key", "mode", "time_signature")
CONTEXT_TYPES = (
    "user_collection",
    "catalog",
    "editorial_playlist",
    "personalized_playlist",
    "radio",
    "charts",
)


def build_track_table(df: pd.DataFrame, min_plays: int = MIN_PLAYS) -> pd.DataFrame:
    """One row per track with >= min_plays plays: outcome, weight, features, covariates."""
    by_track = df.groupby("track_id_clean")
    out = by_track.agg(
        n_plays=("skip_2", "size"),
        skip_rate=("skip_2", "mean"),
        skip_1_rate=("skip_1", "mean"),
        skip_3_rate=("skip_3", "mean"),
        prop_shuffle=("hist_user_behavior_is_shuffle", "mean"),
        prop_premium=("premium", "mean"),
        mean_position=("session_position", "mean"),
    )
    static = by_track[list(AUDIO_FEATURES + TRACK_CONTROLS + TRACK_CATEGORICAL)].first()
    context = pd.crosstab(df["track_id_clean"], df["context_type"], normalize="index")
    context = context.reindex(columns=list(CONTEXT_TYPES), fill_value=0.0)
    context.columns = [f"prop_{c}" for c in context.columns]
    tracks = out.join(static).join(context)
    return tracks[tracks["n_plays"] >= min_plays].reset_index()


def build_session_context_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (session, context type) with that session's skip rate in the context."""
    return (
        df.groupby(["session_id", "context_type"])
        .agg(n_plays=("skip_2", "size"), skip_rate=("skip_2", "mean"))
        .reset_index()
    )


def aggregate(df: pd.DataFrame, out_dir: Path = Path("data/clean")) -> pd.DataFrame:
    """Write both aggregated tables under out_dir and return the track table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks = build_track_table(df)
    tracks.to_csv(out_dir / "tracks.csv", index=False)
    sessions = build_session_context_table(df)
    sessions.to_csv(out_dir / "session_context.csv", index=False)
    print(
        f"aggregate: {len(tracks):,} tracks with >= {MIN_PLAYS} plays "
        f"({int(tracks['n_plays'].sum()):,} plays, "
        f"{tracks['n_plays'].sum() / len(df):.1%} of all plays); "
        f"{len(sessions):,} session x context rows"
    )
    return tracks
