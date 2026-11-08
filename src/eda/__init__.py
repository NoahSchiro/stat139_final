"""Entry point for the EDA."""
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..clean import clean


def load_analysis_table(path: Path) -> pd.DataFrame:
    """Read the cleaned merged session-track table written by clean.py."""
    if not path.exists():
        clean(path)
    df = pd.read_csv(path, parse_dates=["date"])
    df["skip_class"] = pd.Categorical(
        df["skip_class"],
        categories=["completed", "skip_3", "skip_2", "skip_1", "unknown"],
        ordered=True
    )
    return df


def save_fig(fig: plt.Figure, name: str, assets_dir: Path = Path("assets")) -> Path:
    """Save a figure to <assets_dir>/<name>.png."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    path = assets_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")
    return path


def mean_rate(df: pd.DataFrame, by: str | pd.Series, of: str = "skip_2") -> pd.Series:
    """Mean rate of `of` (default skip_2) grouped by column `by` or an aligned Series."""
    return df.groupby(by, observed=True)[of].mean()


def group_counts(df: pd.DataFrame, by: str | pd.Series) -> pd.Series:
    """Row counts grouped by column `by` or an aligned Series."""
    return df.groupby(by, observed=True).size()


def pct(x: float) -> str:
    """Format a proportion as a percentage string."""
    return f"{x:.1%}"

# Having this down here prevents circular imports. Everything above is just helper functions
from src.eda import context, features, outcome, popularity, quality, session

def eda(args) -> None:
    """Run all analyses (or the comma-separated subset in `analyses`),
    printing results and saving figures to assets/."""
    registry: dict[int, Callable[[pd.DataFrame], None]] = {
        1: outcome.skip_flag_structure,
        2: outcome.listening_depth,
        3: session.skip_by_session_position,
        4: session.skip_autocorrelation,
        5: session.session_length,
        6: session.within_session_replays,
        7: session.shuffle_vs_ordered,
        8: context.skip_by_context_type,
        9: context.forward_press_chains,
        10: context.pauses_and_context_switches,
        11: context.premium_vs_free,
        12: context.skip_by_hour_of_day,
        13: features.feature_distributions,
        14: features.feature_skip_correlations,
        15: features.skip_by_duration,
        16: features.duration_context_interaction,
        17: features.feature_collinearity,
        18: popularity.popularity_effect,
        19: popularity.release_recency_effect,
        20: popularity.key_mode_signature,
        21: quality.data_quality_checks,
        22: quality.track_reuse,
    }
    df = load_analysis_table(Path(args.clean_path))
    for number in sorted(registry.keys()):
        run = registry[number]
        question = (run.__doc__ or "").strip()
        print(f"\n=== {number}. {run.__name__}: {question}")
        run(df)
