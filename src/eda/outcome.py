"""EDA on the skip outcome itself: flag structure and listening depth."""
import matplotlib.pyplot as plt
import pandas as pd

from src.eda import pct, save_fig


def skip_flag_structure(df: pd.DataFrame) -> None:
    """How are the four skip flags structured? Are skips nested (as in skip_1 <= skip_2 <= skip_3)?"""
    skip_flags = ["skip_1", "skip_2", "skip_3", "not_skipped"]
    print("  flag rates:", {flag: pct(df[flag].mean()) for flag in skip_flags})
    violations = {
        "skip_1 without skip_2": int((df["skip_1"] & ~df["skip_2"]).sum()),
        "skip_2 without skip_3": int((df["skip_2"] & ~df["skip_3"]).sum()),
        "skip_3 and not_skipped both true": int((df["skip_3"] & df["not_skipped"]).sum()),
    }
    print("  nesting violations:", violations)
    unlabeled = int((~df[list(skip_flags)].any(axis=1)).sum())
    print(f"  plays with no outcome flag set (unlabeled): {unlabeled:,} ({pct(unlabeled / len(df))})")
    flags = list(skip_flags)
    patterns = df.groupby(flags, observed=True).size()
    print("  observed (skip_1, skip_2, skip_3, not_skipped) patterns:")
    for pattern, count in patterns.items():
        print(f"    {pattern}: {count:,}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(skip_flags, [df[flag].mean() for flag in skip_flags])
    ax.set_ylabel("share of plays")
    ax.set_title("Rate of each skip flag")
    save_fig(fig, "01_skip_flag_rates")


def listening_depth(df: pd.DataFrame) -> None:
    """How deep into a track do listeners get, i.e. what is the distribution of skip_class?"""
    dist = df["skip_class"].value_counts().sort_index()
    print("  skip_class distribution:", {str(k): pct(v / len(df)) for k, v in dist.items()})

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(k) for k in dist.index], dist.values / len(df))
    ax.set_ylabel("share of plays")
    ax.set_title("Listening depth per play (most severe skip flag; completed = no skip)")
    save_fig(fig, "02_skip_class_distribution")
