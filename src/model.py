"""Baseline model: weighted least squares of track skip rate on audio features.

Unit: one row per track with at least five plays. Weights are the play count, since a
rate estimated from more plays has smaller variance. Audio features and controls are
standardized so coefficients are comparable (change in skip rate per 1 SD). The main
hypothesis is tested with a nested F-test: covariates-only versus covariates plus the
audio-feature block.

Figures and CSV tables are written to assets/. The report copies the numbers it needs
from those outputs; nothing here generates report source.
"""

from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.outliers_influence import (
    OLSInfluence,
    variance_inflation_factor,
)

from src.aggregate import AUDIO_FEATURES, aggregate
from src.eda import load_analysis_table, save_fig

ASSETS = Path("assets")


def design(tracks: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Standardized design matrix (plus intercept) for the given columns."""
    x = tracks[list(columns)].astype(float)
    x = (x - x.mean()) / x.std()
    x.insert(0, "const", 1.0)
    return x


def fit_wls(tracks: pd.DataFrame, columns: tuple[str, ...]) -> RegressionResultsWrapper:
    """WLS of skip_rate on standardized columns, weighted by play count."""
    model = sm.WLS(
        tracks["skip_rate"], design(tracks, columns), weights=tracks["n_plays"]
    )
    return model.fit()


def coefficient_table(fit: RegressionResultsWrapper, path: Path) -> pd.DataFrame:
    """Save the coefficient table as CSV (also returned, in display order)."""
    table = pd.DataFrame(
        {
            "coef": fit.params,
            "se": fit.bse,
            "t": fit.tvalues,
            "p": fit.pvalues,
            "lo": fit.conf_int()[0],
            "hi": fit.conf_int()[1],
        }
    ).drop(index="const")
    table.round(4).to_csv(path)
    return table


def vif_table(x: pd.DataFrame, path: Path) -> pd.Series:
    """Variance inflation factors for each non-intercept column."""
    cols = [c for c in x.columns if c != "const"]
    vif = pd.Series(
        [variance_inflation_factor(x.to_numpy(), x.columns.get_loc(c)) for c in cols],
        index=cols,
    ).sort_values(ascending=False)
    vif.round(1).rename("vif").to_csv(path)
    return vif


def plot_coefficients(table: pd.DataFrame, name: str) -> None:
    """Dot-and-interval plot of the standardized coefficients."""
    table = table.sort_values("coef")
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["C3" if f in AUDIO_FEATURES else "C0" for f in table.index]
    ax.hlines(range(len(table)), table["lo"], table["hi"], colors=colors, lw=1.5)
    ax.scatter(table["coef"], range(len(table)), c=colors, zorder=3)
    ax.set_yticks(range(len(table)), table.index)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("change in skip rate per 1 SD of predictor (95% CI)")
    ax.set_title("Baseline WLS coefficients (red: audio features, blue: covariates)")
    save_fig(fig, name)


def plot_diagnostics(
    fit: RegressionResultsWrapper, tracks: pd.DataFrame, name: str
) -> None:
    """Residuals vs fitted, QQ plot of weighted residuals, and Cook's distance."""
    fitted = np.asarray(fit.fittedvalues)
    resid = np.asarray(fit.resid) * np.sqrt(tracks["n_plays"].to_numpy())
    cooks = OLSInfluence(fit).cooks_distance[0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].scatter(fitted, resid, s=4, alpha=0.3)
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].set_xlabel("fitted skip rate")
    axes[0].set_ylabel("weighted residual")
    axes[0].set_title("Residuals vs fitted")
    stats.probplot(resid, plot=axes[1])
    axes[1].set_title("Normal QQ plot of weighted residuals")
    axes[2].stem(cooks, markerfmt=" ", basefmt=" ")
    axes[2].set_xlabel("track index")
    axes[2].set_ylabel("Cook's distance")
    axes[2].set_title("Influence")
    save_fig(fig, name)


def model(args: Namespace) -> None:
    """Fit the baseline model, print the summary, and save figures and tables."""
    df = load_analysis_table(Path(args.clean_path))
    tracks = aggregate(df).dropna(subset=list(AUDIO_FEATURES))
    print(f"model: {len(tracks):,} tracks after dropping rows with masked features")

    # Covariates named in the proposal: track-level "trendiness" controls, then the
    # context mix aggregated from each track's plays. user_collection is the omitted
    # (reference) context so the six context shares, which sum to one, are not collinear.
    covariates = (
        "us_popularity_estimate",
        "release_year",
        "prop_catalog",
        "prop_editorial_playlist",
        "prop_personalized_playlist",
        "prop_radio",
        "prop_charts",
        "prop_shuffle",
        "prop_premium",
        "mean_position",
    )
    reduced = fit_wls(tracks, covariates)
    full = fit_wls(tracks, covariates + AUDIO_FEATURES)
    f_stat, p_value, df_diff = full.compare_f_test(reduced)
    print(full.summary())
    print(
        f"\nnested F-test (audio block, {int(df_diff)} df): F = {f_stat:.2f}, p = {p_value:.2e}"
    )
    print(
        f"R^2 covariates only {reduced.rsquared:.4f}; with audio {full.rsquared:.4f} "
        f"(adj. {full.rsquared_adj:.4f})"
    )

    x = design(tracks, covariates + AUDIO_FEATURES)
    vif = vif_table(x, ASSETS / "baseline_vif.csv")
    print("VIF > 5:", {k: round(v, 1) for k, v in vif[vif > 5].items()})

    table = coefficient_table(full, ASSETS / "baseline_coefficients.csv")
    print("\ncoefficients (standardized):")
    print(table.round(4).to_string())
    plot_coefficients(table, "28_baseline_coefficients")
    plot_diagnostics(full, tracks, "29_baseline_diagnostics")
