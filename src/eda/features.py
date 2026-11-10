"""EDA on track audio features: distributions, univariate effects, duration, interactions, collinearity (analyses 13-17)."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda import pct, save_fig

FEATURES = (
    "duration",
    "release_year",
    "tempo",
    "danceability",
    "energy",
    "valence",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "beat_strength",
)


def feature_distributions(df: pd.DataFrame) -> None:
    """What do the marginal distributions of the audio features look like (skew, degenerate mass)?"""
    skew = df[list(FEATURES)].skew().sort_values()
    print(
        "  most negatively skewed:", {f: round(v, 2) for f, v in skew.head(3).items()}
    )
    print(
        "  most positively skewed:", {f: round(v, 2) for f, v in skew.tail(3).items()}
    )
    print(
        f"  plays with instrumentalness < 0.01: {pct((df['instrumentalness'] < 0.01).mean())}"
    )
    print(
        f"  plays with missing tempo (masked in cleaning): {pct(df['tempo'].isna().mean())}"
    )

    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    for ax, feature in zip(axes.flat, FEATURES):
        ax.hist(df[feature].dropna(), bins=40)
        ax.set_title(feature)
    fig.suptitle("Audio feature distributions")
    save_fig(fig, "13_feature_distributions")


def feature_skip_correlations(df: pd.DataFrame) -> None:
    """Which numeric features correlate with skipping?"""
    numeric = df.select_dtypes("number").drop(
        columns=["session_position", "session_length"]
    )
    correlations = numeric.corrwith(df["skip_2"].astype(float)).sort_values()
    print(
        "  strongest negative:",
        {f: round(v, 4) for f, v in correlations.head(5).items()},
    )
    print(
        "  strongest positive:",
        {f: round(v, 4) for f, v in correlations.tail(5).items()},
    )

    fig, ax = plt.subplots(figsize=(7, 10))
    ax.barh(correlations.index, correlations.to_numpy())
    ax.set_xlabel("Pearson correlation with skip_2")
    ax.set_title("Univariate correlations with skipping")
    ax.axvline(0, color="gray", lw=0.8)
    save_fig(fig, "14_feature_correlations")


def skip_by_duration(df: pd.DataFrame) -> None:
    """Does skip probability rise monotonically with track duration?"""
    bins = pd.qcut(df["duration"], q=10)
    rate = df.groupby(bins, observed=True)["skip_2"].mean()
    labels = [f"{int(interval.left)}-{int(interval.right)}s" for interval in rate.index]
    print(
        f"  shortest bin {labels[0]}: {pct(rate.iloc[0])}; longest bin {labels[-1]}: {pct(rate.iloc[-1])}"
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(rate)), rate.to_numpy())
    ax.set_xticks(range(len(rate)), labels, rotation=30)
    ax.set_ylabel("skip_2 rate")
    ax.set_title("Skip rate by duration decile")
    save_fig(fig, "15_skip_rate_by_duration")


def duration_context_interaction(df: pd.DataFrame) -> None:
    """Does the duration-skip relationship persist within each context type?"""
    bins = pd.qcut(df["duration"], q=5)
    grid = df.groupby([bins, "context_type"], observed=True)["skip_2"].mean().unstack()
    spread = grid.max() - grid.min()
    print("  rate spread across duration quintiles within each context:")
    print(spread.map(pct).to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(grid.values, cmap="coolwarm", vmin=0.35, vmax=0.7, aspect="auto")
    ax.set_xticks(range(grid.shape[1]), grid.columns, rotation=25)
    ax.set_yticks(
        range(grid.shape[0]), [f"{int(i.left)}-{int(i.right)}s" for i in grid.index]
    )
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(
                j, i, f"{grid.values[i, j]:.2f}", ha="center", va="center", fontsize=9
            )
    fig.colorbar(image, ax=ax, label="skip_2 rate")
    ax.set_title("Skip rate: duration quintile x context type")
    save_fig(fig, "16_duration_by_context")


def feature_collinearity(df: pd.DataFrame) -> None:
    """How correlated are the audio features with each other?"""
    corr = df[list(FEATURES)].corr()
    upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = cast(pd.Series, corr.where(upper).stack())
    pairs = pairs.loc[pairs.abs().sort_values(ascending=False).index]
    strong = pairs[pairs.abs() > 0.5]
    print("  |r| > 0.5 pairs:")
    print(strong.round(3).to_string() if len(strong) else "    none")

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=45, ha="right")
    ax.set_yticks(range(len(FEATURES)), FEATURES)
    fig.colorbar(image, ax=ax, label="Pearson r")
    ax.set_title("Audio feature correlation matrix")
    save_fig(fig, "17_feature_collinearity")
