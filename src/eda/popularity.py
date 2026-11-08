"""EDA on track popularity, recency, and musical attributes."""
import matplotlib.pyplot as plt
import pandas as pd

from src.eda import pct, save_fig


def popularity_effect(df: pd.DataFrame) -> None:
    """Is us_popularity_estimate informative within its [90, 100] range?"""
    popularity = df["us_popularity_estimate"]
    bins = pd.qcut(popularity, q=10, duplicates="drop")
    rate = df.groupby(bins, observed=True)["skip_2"].mean()
    print(f"  popularity range: [{popularity.min():.2f}, {popularity.max():.2f}]")
    print(f"  correlation with skip_2: {popularity.corr(df['skip_2']):.4f}")
    print(f"  skip rate, bottom vs top decile: {pct(rate.iloc[0])} vs {pct(rate.iloc[-1])}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(popularity, bins=40)
    axes[0].set_xlabel("us_popularity_estimate")
    axes[0].set_ylabel("plays")
    axes[0].set_title("Popularity distribution")
    labels = [f"{interval.left:.2f}" for interval in rate.index]
    axes[1].bar(range(len(rate)), rate.values)
    axes[1].set_xticks(range(len(rate)), labels, rotation=30)
    axes[1].set_ylabel("skip_2 rate")
    axes[1].set_title("Skip rate by popularity decile")
    save_fig(fig, "18_popularity")


def release_recency_effect(df: pd.DataFrame) -> None:
    """Does track recency matter: how is release_year distributed? Do older tracks skip differently?"""
    decade = df["release_year"] // 10 * 10
    stats = df.groupby(decade)["skip_2"].agg(rate="mean", n="size")
    print(f"  plays of pre-2000 tracks: {pct((df['release_year'] < 2000).mean())}")
    print(stats.assign(rate=stats["rate"].map(pct)).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["release_year"], bins=40)
    axes[0].set_xlabel("release_year")
    axes[0].set_ylabel("plays")
    axes[0].set_title("Release year distribution")
    axes[1].bar([str(int(d)) for d in stats.index], stats["rate"])
    axes[1].set_ylabel("skip_2 rate")
    axes[1].set_title("Skip rate by release decade")
    axes[1].tick_params(axis="x", rotation=45)
    save_fig(fig, "19_release_year")


def key_mode_signature(df: pd.DataFrame) -> None:
    """Do musical key, mode (major/minor), or time signature predict skipping?"""
    by_key = df.groupby("key")["skip_2"].mean()
    by_mode = df.groupby("mode", observed=True)["skip_2"].mean()
    by_signature = df.groupby("time_signature", observed=True)["skip_2"].mean()
    print(f"  key rate spread: {pct(by_key.max() - by_key.min())} (max {pct(by_key.max())}, min {pct(by_key.min())})")
    print(f"  mode rates: {', '.join(f'{m}: {pct(r)}' for m, r in by_mode.items())}")
    print(f"  time_signature rates: {', '.join(f'{s}: {pct(r)} (n={int((df['time_signature'] == s).sum()):,})' for s, r in by_signature.items())}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(by_key.index, by_key.values)
    axes[0].set_xlabel("key (pitch class)")
    axes[0].set_ylabel("skip_2 rate")
    axes[0].set_title("Skip rate by key")
    axes[1].bar([str(m) for m in by_mode.index], by_mode.values)
    axes[1].set_ylabel("skip_2 rate")
    axes[1].set_title("Skip rate by mode")
    save_fig(fig, "20_key_and_mode")
