"""Baseline model: weighted least squares of track skip rate on audio features.

Unit: one row per track with >= MIN_PLAYS plays. Weights are the play count, since a
rate estimated from more plays has smaller variance. Audio features and controls are
standardized so coefficients are comparable (change in skip rate per 1 SD). The main
hypothesis is tested with a nested F-test: covariates-only versus covariates plus the
audio-feature block.

Outputs go to assets/ (figures) and reports/eda/tables/ (Typst table fragments).
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

# user_collection is the omitted (reference) context so the shares are not collinear.
COVARIATES = (
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
TABLE_DIR = Path("reports/eda/tables")


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


def typst_rows(rows: list[list[str]], path: Path) -> None:
    """Write rows as a Typst array literal that main.typ splices into a table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ",\n".join("  (" + ", ".join(f'"{c}"' for c in row) + ")" for row in rows)
    path.write_text(f"(\n{body},\n)\n")


def coefficient_table(fit: RegressionResultsWrapper, path: Path) -> pd.DataFrame:
    """Save the coefficient table (also returned, in display order)."""
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
    rows = [
        [str(name), f"{r.coef:+.4f}", f"{r.se:.4f}", f"{r.t:+.2f}", f"{r.p:.3f}"]
        for name, r in table.iterrows()
    ]
    typst_rows(rows, path)
    return table


def vif_table(x: pd.DataFrame, path: Path) -> pd.Series:
    """Variance inflation factors for each non-intercept column."""
    cols = [c for c in x.columns if c != "const"]
    vif = pd.Series(
        [variance_inflation_factor(x.to_numpy(), x.columns.get_loc(c)) for c in cols],
        index=cols,
    ).sort_values(ascending=False)
    typst_rows([[str(name), f"{v:.1f}"] for name, v in vif.items()], path)
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

    reduced = fit_wls(tracks, COVARIATES)
    full = fit_wls(tracks, COVARIATES + AUDIO_FEATURES)
    f_stat, p_value, df_diff = full.compare_f_test(reduced)
    print(full.summary())
    print(
        f"\nnested F-test (audio block, {int(df_diff)} df): F = {f_stat:.2f}, p = {p_value:.2e}"
    )
    print(
        f"R^2 covariates only {reduced.rsquared:.4f}; with audio {full.rsquared:.4f} "
        f"(adj. {full.rsquared_adj:.4f})"
    )

    x = design(tracks, COVARIATES + AUDIO_FEATURES)
    vif = vif_table(x, TABLE_DIR / "vif.typ")
    print("VIF > 5:", {k: round(v, 1) for k, v in vif[vif > 5].items()})

    table = coefficient_table(full, TABLE_DIR / "coefficients.typ")
    typst_rows(
        [
            ["Tracks (n)", f"{int(full.nobs):,}"],
            ["Plays covered", f"{int(tracks['n_plays'].sum()):,}"],
            ["R² (covariates only)", f"{reduced.rsquared:.3f}"],
            ["R² (covariates + audio)", f"{full.rsquared:.3f}"],
            ["Adjusted R² (full)", f"{full.rsquared_adj:.3f}"],
            [
                "Nested F (audio block)",
                f"{f_stat:.2f} on {int(df_diff)} and {int(full.df_resid)} df",
            ],
            ["Nested F p-value", f"{p_value:.1e}"],
        ],
        TABLE_DIR / "fit.typ",
    )
    plot_coefficients(table, "28_baseline_coefficients")
    plot_diagnostics(full, tracks, "29_baseline_diagnostics")
