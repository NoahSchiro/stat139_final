"""EDA on within-session structure: position, autocorrelation, length, repeats, shuffle."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda import mean_rate, pct, save_fig


def skip_by_session_position(df: pd.DataFrame) -> None:
    """Does skip probability depend on position within the session?"""
    rate = mean_rate(df, "session_position")
    print(
        f"  position 1 rate {pct(rate.iloc[0])} vs overall {pct(df['skip_2'].mean())}; "
        f"mean of positions 2-20 {pct(rate.iloc[1:].mean())}"
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(rate.index, rate.values)
    ax.axhline(df["skip_2"].mean(), color="gray", ls="--", label="overall rate")
    ax.set_xlabel("session_position")
    ax.set_ylabel("skip_2 rate")
    ax.set_title("Skip rate by position within the session")
    ax.legend()
    save_fig(fig, "03_skip_rate_by_position")


def skip_autocorrelation(df: pd.DataFrame) -> None:
    """Is skipping autocorrelated within a session, i.e. P(skip at t+1 | skip at t)?"""
    ordered = df.sort_values(["session_id", "session_position"], kind="stable")
    prev_skip = ordered.groupby("session_id")["skip_2"].shift()
    has_prev = prev_skip.notna()
    conditional = ordered[has_prev].groupby(prev_skip[has_prev].astype(bool))["skip_2"].mean()
    overall = df["skip_2"].mean()
    print(
        f"  P(skip | previous skipped) {pct(conditional[True])} vs "
        f"P(skip | previous played) {pct(conditional[False])} (overall {pct(overall)})"
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["previous played", "previous skipped"], [conditional[False], conditional[True]])
    ax.axhline(overall, color="gray", ls="--", label="overall rate")
    ax.set_ylabel("skip_2 rate for next track")
    ax.set_title("Skip persistence within a session")
    ax.legend()
    save_fig(fig, "04_skip_autocorrelation")


def session_length(df: pd.DataFrame) -> None:
    """How long do sessions run, is the length capped, and does length relate to skipping?"""
    sessions = df.groupby("session_id").agg(
        length=("session_length", "first"),
        skip_rate=("skip_2", "mean"),
    )
    print(f"  sessions at the 20-track cap: {pct((sessions['length'] == 20).mean())}")
    print(f"  sessions at the 10-track minimum: {pct((sessions['length'] == 10).mean())}")
    print(f"  per-session skip rate vs length correlation: {sessions['length'].corr(sessions['skip_rate']):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.arange(9.5, 21.5)
    axes[0].hist(sessions["length"], bins=bins, rwidth=0.9)
    axes[0].set_xlabel("session_length")
    axes[0].set_ylabel("sessions")
    axes[0].set_title("Session length distribution")
    by_length = df.groupby("session_length", observed=True)["skip_2"].mean()
    axes[1].plot(by_length.index, by_length.values, marker="o")
    axes[1].set_xlabel("session_length")
    axes[1].set_ylabel("skip_2 rate")
    axes[1].set_title("Skip rate by session length")
    save_fig(fig, "05_session_length")


def within_session_replays(df: pd.DataFrame) -> None:
    """Do people replay tracks within a session, and are replays skipped less?"""
    ordered = df.sort_values(["session_id", "session_position"], kind="stable")
    is_first_play = ordered.groupby(["session_id", "track_id_clean"]).cumcount().eq(0)
    play_counts = ordered.groupby(["session_id", "track_id_clean"], observed=True).size()
    dist = play_counts.value_counts().sort_index()

    rate = ordered.groupby(is_first_play)["skip_2"].mean()
    print(f"  plays that are within-session repeats: {pct((~is_first_play).mean())}")
    print(f"  skip rate of first plays {pct(rate[True])} vs repeat plays {pct(rate[False])}")
    print(f"  max plays of one track within a session: {int(play_counts.max())}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar([str(n) for n in dist.index], dist.values)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("plays of one track within a session")
    axes[0].set_ylabel("(session, track) pairs (log)")
    axes[0].set_title("Within-session replay counts")
    axes[1].bar(["first play", "repeat play"], [rate[True], rate[False]])
    axes[1].set_ylabel("skip_2 rate")
    axes[1].set_title("Skip rate: first vs repeat plays")
    save_fig(fig, "06_within_session_repeats")


def shuffle_vs_ordered(df: pd.DataFrame) -> None:
    """Do shuffle-mode sessions skip differently than sequentially ordered listening?"""
    rate = df.groupby("hist_user_behavior_is_shuffle")["skip_2"].mean()
    n = df.groupby("hist_user_behavior_is_shuffle").size()
    print(f"  shuffle plays: {int(n[True]):,} ({pct(n[True] / len(df))}); skip rate {pct(rate[True])} vs {pct(rate[False])}")
    by_context = df.groupby(["context_type", "hist_user_behavior_is_shuffle"], observed=True)["skip_2"].mean().unstack()
    print("  skip rate by context x shuffle:")
    print(by_context.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(["ordered", "shuffle"], [rate[False], rate[True]])
    axes[0].set_ylabel("skip_2 rate")
    axes[0].set_title("Skip rate: shuffle vs ordered")
    by_context.plot(kind="bar", ax=axes[1])
    axes[1].set_ylabel("skip_2 rate")
    axes[1].set_title("Skip rate by context x shuffle")
    axes[1].tick_params(axis="x", rotation=30)
    save_fig(fig, "07_shuffle_effect")
