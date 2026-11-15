"""EDA at the proposal's analysis units: tracks with >= 5 plays and session x context (analyses 23-27)."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.aggregate import AUDIO_FEATURES, build_track_table
from src.eda import pct, save_fig


def plays_per_track(df: pd.DataFrame) -> None:
    """How sparse is track coverage, and how much data survives the minimum-play threshold?"""
    n = df.groupby("track_id_clean").size()
    for k in (1, 5, 10, 20):
        kept = n[n >= k]
        print(
            f"  >= {k:>2} plays: {len(kept):>6,} tracks, {int(kept.sum()):>7,} plays "
            f"({pct(kept.sum() / len(df))} of plays)"
        )

    fig, ax = plt.subplots(figsize=(8, 4))
    edges = np.logspace(0, np.log10(n.max() + 1), 40).tolist()
    ax.hist(n, bins=edges)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for k, style in ((5, "-"), (10, "--"), (20, ":")):
        ax.axvline(k, color="gray", ls=style, label=f">= {k} plays")
    ax.set_xlabel("plays of the track in the sample (log)")
    ax.set_ylabel("tracks (log)")
    ax.set_title("Plays per track and candidate minimum-play thresholds")
    ax.legend()
    save_fig(fig, "23_plays_per_track")


def track_skip_rate_distribution(df: pd.DataFrame) -> None:
    """What does the track-level skip rate (the regression outcome) look like?"""
    tracks = build_track_table(df)
    rate = tracks["skip_rate"]
    print(
        f"  mean {rate.mean():.3f}, median {rate.median():.3f}, SD {rate.std():.3f}; "
        f"tracks at exactly 0: {pct((rate == 0).mean())}, at exactly 1: {pct((rate == 1).mean())}"
    )
    for lo, hi in ((5, 9), (10, 19), (20, 10_000)):
        sub = rate[tracks["n_plays"].between(lo, hi)]
        print(
            f"  plays {lo}-{hi}: {len(sub):,} tracks, SD of skip rate {sub.std():.3f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(rate, bins=30)
    axes[0].axvline(rate.mean(), color="gray", ls="--", label=f"mean {rate.mean():.2f}")
    axes[0].set_xlabel("track skip rate (share of plays with skip_2)")
    axes[0].set_ylabel("tracks")
    axes[0].set_title(
        f"Skip rate across {len(tracks):,} tracks with >= {int(tracks['n_plays'].min())} plays"
    )
    axes[0].legend()
    axes[1].scatter(tracks["n_plays"], rate, s=4, alpha=0.3)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("plays (log)")
    axes[1].set_ylabel("track skip rate")
    axes[1].set_title("Skip rate variance shrinks with play count")
    save_fig(fig, "24_track_skip_rate")


def track_skip_vs_features(df: pd.DataFrame) -> None:
    """At the track level, how does skip rate move with each audio feature?"""
    tracks = build_track_table(df)
    fig, axes = plt.subplots(4, 4, figsize=(16, 13))
    for ax, feature in zip(axes.flat, AUDIO_FEATURES):
        sub = tracks[[feature, "skip_rate", "n_plays"]].dropna()
        bins = pd.qcut(sub[feature], q=10, duplicates="drop")
        grouped = sub.groupby(bins, observed=True)
        x = grouped[feature].median()
        y = grouped["skip_rate"].mean()
        se: pd.Series = grouped["skip_rate"].std() / np.sqrt(grouped.size())
        ax.scatter(sub[feature], sub["skip_rate"], s=2, alpha=0.12, color="gray")
        ax.errorbar(x, y, yerr=1.96 * se, fmt="o-", color="C3", ms=4, capsize=2)
        ax.set_title(feature)
        ax.set_ylim(0.3, 0.75)
    fig.suptitle("Track skip rate by audio feature (decile means with 95% CI)")
    save_fig(fig, "25_track_skip_vs_features")


def track_feature_correlations(df: pd.DataFrame) -> None:
    """Which audio features correlate with skip rate once plays are aggregated to tracks?"""
    tracks = build_track_table(df)
    cols = list(AUDIO_FEATURES) + ["us_popularity_estimate", "release_year"]
    corr = tracks[cols].corrwith(tracks["skip_rate"]).sort_values()
    play_corr = df[cols].corrwith(df["skip_2"].astype(float))
    print("  track-level r (play-level r in parentheses):")
    for f in corr.index:
        print(f"    {f:<24} {corr[f]:+.3f}  ({play_corr[f]:+.3f})")

    pair = tracks[list(AUDIO_FEATURES)].corr()
    upper = pair.where(np.triu(np.ones(pair.shape, dtype=bool), k=1))
    strong = cast(pd.Series, upper.stack())
    strong = strong[strong.abs() > 0.5]
    strong = strong.loc[strong.abs().sort_values(ascending=False).index]
    print("  track-level feature pairs with |r| > 0.5:")
    print(strong.round(3).to_string() if len(strong) else "    none")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), width_ratios=(1, 1.4))
    axes[0].barh(corr.index, corr.to_numpy())
    axes[0].axvline(0, color="gray", lw=0.8)
    axes[0].set_xlabel("Pearson r with track skip rate")
    axes[0].set_title("Univariate correlations (track level)")
    image = axes[1].imshow(pair.values, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_xticks(
        range(len(AUDIO_FEATURES)), AUDIO_FEATURES, rotation=45, ha="right"
    )
    axes[1].set_yticks(range(len(AUDIO_FEATURES)), AUDIO_FEATURES)
    axes[1].set_title("Feature correlation matrix (track level)")
    fig.colorbar(image, ax=axes[1], label="Pearson r", shrink=0.8)
    save_fig(fig, "26_track_feature_correlations")


def session_skip_by_context(df: pd.DataFrame) -> None:
    """How is per-session skip rate distributed within each listening context (the ANOVA unit)?"""
    sessions = (
        df.groupby(["session_id", "context_type"])["skip_2"]
        .agg(["mean", "size"])
        .reset_index()
    )
    sessions = sessions[sessions["size"] >= 3]
    print(f"  session x context rows with >= 3 plays: {len(sessions):,}")
    stats = sessions.groupby("context_type")["mean"].agg(["size", "mean", "std"])
    print(stats.round(3).to_string())

    order = list(stats.sort_values("mean").index)
    data = [sessions.loc[sessions["context_type"] == c, "mean"] for c in order]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.boxplot(data, tick_labels=order, showmeans=True)
    ax.axhline(df["skip_2"].mean(), color="gray", ls="--", lw=0.8, label="overall rate")
    ax.set_ylabel("session skip rate within context")
    ax.set_title(
        "Per-session skip rate by listening context (sessions with >= 3 plays in context)"
    )
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    save_fig(fig, "27_session_skip_by_context")
