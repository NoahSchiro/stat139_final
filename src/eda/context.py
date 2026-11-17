"""EDA on listening context and behavior: context type, skip chains, pauses, premium, time of day."""

from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

from src.eda import group_counts, mean_rate, pct, save_fig


def skip_by_context_type(df: pd.DataFrame) -> None:
    """How does skip rate vary by context_type? Why do algorithmic contexts skip more?"""
    stats = (
        df.groupby("context_type", observed=True)
        .agg(rate=("skip_2", "mean"), n=("skip_2", "size"))
        .sort_values("rate")
    )
    print(stats.assign(rate=stats["rate"].map(pct)).to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(stats.index, stats["rate"])
    ax.axhline(df["skip_2"].mean(), color="gray", ls="--", label="overall rate")
    ax.set_ylabel("skip_2 rate")
    ax.set_title("Skip rate by context type")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    save_fig(fig, "08_skip_rate_by_context")


def forward_press_chains(df: pd.DataFrame) -> None:
    """How long do consecutive forward-press chains run within a session?"""
    ordered = df.sort_values(["session_id", "session_position"], kind="stable")
    fwd = ordered["hist_user_behavior_reason_start"].eq("fwdbtn")
    new_run = fwd.ne(fwd.shift()) | ordered["session_id"].ne(
        ordered["session_id"].shift()
    )
    run_id = new_run.cumsum()
    run_sizes = fwd.groupby(run_id).sum()
    fwd_runs = run_sizes[run_sizes > 0]

    print(f"  plays started by a forward press: {pct(fwd.mean())}")
    print(
        f"  forward-run lengths: mean {fwd_runs.mean():.2f}, max {int(fwd_runs.max())}"
    )
    print(
        f"  forward-press plays inside runs of 3+: {pct((fwd_runs[fwd_runs >= 3].sum()) / fwd_runs.sum())}"
    )

    dist = fwd_runs.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(dist.index, dist.to_numpy())
    ax.set_xlabel("consecutive forward-presses in a row")
    ax.set_ylabel("runs")
    ax.set_title("Distribution of forward-press chain lengths")
    save_fig(fig, "09_skip_chain_lengths")


def pauses_and_context_switches(df: pd.DataFrame) -> None:
    """What is the relationship between pre-play pauses, context switches, and skipping?"""

    pause_cols = [
        "no_pause_before_play",
        "short_pause_before_play",
        "long_pause_before_play",
    ]

    def pause_label(row: pd.Series) -> str:
        return (
            "+".join(
                name.replace("_before_play", "") for name in pause_cols if row[name]
            )
            or "none"
        )

    labels = df[pause_cols].apply(pause_label, axis=1)
    stats = (
        df.groupby(labels, observed=True)
        .agg(rate=("skip_2", "mean"), n=("skip_2", "size"))
        .sort_values("rate", ascending=False)
    )
    print(stats.assign(rate=stats["rate"].map(pct)).to_string())
    print(
        f"  overlap: no_pause AND long_pause both true: {int((df['no_pause_before_play'] & df['long_pause_before_play']).sum()):,}"
    )
    session_start = df["session_position"].eq(1)
    print(
        f"  'none' pause combo is exactly the session-start plays: {labels.eq('none').equals(session_start)}"
    )
    print(
        f"  skip rate after a context switch: {pct(df.loc[df['context_switch'], 'skip_2'].mean())}"
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(stats.index, stats["rate"])
    ax.set_ylabel("skip_2 rate")
    ax.set_title("Skip rate by pre-play pause combination")
    ax.tick_params(axis="x", rotation=25)
    save_fig(fig, "10_skip_rate_by_pause")


def premium_vs_free(df: pd.DataFrame) -> None:
    """Do premium and free users skip differently, and does their context mix explain it?"""
    rate = df.groupby("premium")["skip_2"].mean()
    mix = pd.crosstab(df["context_type"], df["premium"], normalize="columns")
    print(f"  skip rate: premium {pct(rate[True])} vs free {pct(rate[False])}")
    print("  context mix (column shares):")
    print((mix * 100).round(1).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(["free", "premium"], [rate[False], rate[True]])
    axes[0].set_ylabel("skip_2 rate")
    axes[0].set_title("Skip rate by account type")
    mix.plot(kind="bar", ax=axes[1])
    axes[1].set_ylabel("share of plays")
    axes[1].set_title("Context mix by account type")
    axes[1].tick_params(axis="x", rotation=25)
    save_fig(fig, "11_premium_vs_free")


def skip_by_hour_of_day(df: pd.DataFrame) -> None:
    """Is there an hour-of-day effect on skipping? Does play volume confound it?"""
    rate = mean_rate(df, "hour_of_day")
    plays = group_counts(df, "hour_of_day")
    low_hour, high_hour = cast(int, rate.idxmin()), cast(int, rate.idxmax())
    print(
        f"  lowest skip hour {int(low_hour)}:00 ({pct(rate.min())}); highest {int(high_hour)}:00 ({pct(rate.max())})"
    )
    print(
        f"  quietest hour {int(cast(int, plays.idxmin()))}:00 ({int(plays.min()):,} plays); busiest {int(cast(int, plays.idxmax()))}:00 ({int(plays.max()):,})"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(plays.index, plays.values)
    axes[0].set_xlabel("hour of day")
    axes[0].set_ylabel("plays")
    axes[0].set_title("Play volume by hour")
    axes[1].plot(rate.index, rate.values, marker="o")
    axes[1].axhline(df["skip_2"].mean(), color="gray", ls="--", label="overall rate")
    axes[1].set_xlabel("hour of day")
    axes[1].set_ylabel("skip_2 rate")
    axes[1].set_title("Skip rate by hour")
    axes[1].legend()
    save_fig(fig, "12_hour_of_day")
