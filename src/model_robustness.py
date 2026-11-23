from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.aggregate import AUDIO_FEATURES
from src.model import design
from src.model_analysis import CONTROLS

ASSETS = Path("assets")

# Model B: keep bounciness as the rhythm feature
REMOVE = {
    "beat_strength",
    "danceability",
    "dyn_range_mean",
    "mechanism",
    "organism",
}

AUDIO = tuple(f for f in AUDIO_FEATURES if f not in REMOVE)
PREDICTORS = CONTROLS + AUDIO
OUTCOMES = ("skip_1", "skip_2", "skip_3")


def prepare_data():
    df = pd.read_csv("data/clean/session_tracks.csv")

    boolean_cols = list(OUTCOMES) + ["hist_user_behavior_is_shuffle", "premium"]

    for col in boolean_cols:
        df[col] = (
            df[col].astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
        )

    if df[boolean_cols].isna().any().any():
        raise ValueError("Unexpected values in boolean columns")

    track_cols = list(AUDIO) + ["us_popularity_estimate", "release_year"]

    agg = {col: (col, "first") for col in track_cols}
    agg.update(
        {
            "n_plays": ("skip_2", "size"),
            "skip_1_rate": ("skip_1", "mean"),
            "skip_2_rate": ("skip_2", "mean"),
            "skip_3_rate": ("skip_3", "mean"),
            "prop_shuffle": ("hist_user_behavior_is_shuffle", "mean"),
            "prop_premium": ("premium", "mean"),
            "mean_position": ("session_position", "mean"),
        }
    )

    tracks = df.groupby("track_id_clean").agg(**agg)

    # Context proportions, with user_collection as the reference
    shares = pd.crosstab(df["track_id_clean"], df["context_type"], normalize="index")
    contexts = (
        "catalog",
        "editorial_playlist",
        "personalized_playlist",
        "radio",
        "charts",
    )

    for context in contexts:
        tracks[f"prop_{context}"] = shares.get(context, 0.0)

    return tracks


def fit_model(data, outcome):
    x = design(data, PREDICTORS)
    return sm.WLS(data[outcome], x, weights=data["n_plays"]).fit(cov_type="HC3")


def main():
    ASSETS.mkdir(exist_ok=True)
    tracks = prepare_data()

    results = []

    # Step 1. Alternative outcomes and minimum-play thresholds
    for threshold in (5, 10, 20):
        for skip in OUTCOMES:
            outcome = f"{skip}_rate"

            subset = tracks.loc[tracks["n_plays"] >= threshold].dropna(
                subset=list(PREDICTORS) + [outcome]
            )
            fit = fit_model(subset, outcome)

            # Joint HC3-robust Wald test for audio features
            indices = [fit.params.index.get_loc(col) for col in AUDIO]
            R = np.eye(len(fit.params))[indices]
            test = fit.wald_test(R, scalar=True)

            results.append(
                {
                    "outcome": skip,
                    "min_plays": threshold,
                    "n_tracks": len(subset),
                    "R2": fit.rsquared,
                    "audio_block_p": float(test.pvalue),
                    "bounciness_coef": fit.params["bounciness"],
                    "bounciness_p": fit.pvalues["bounciness"],
                    "duration_coef": fit.params["duration"],
                    "duration_p": fit.pvalues["duration"],
                    "speechiness_coef": fit.params["speechiness"],
                    "speechiness_p": fit.pvalues["speechiness"],
                    "loudness_coef": fit.params["loudness"],
                    "loudness_p": fit.pvalues["loudness"],
                }
            )

    comparison = pd.DataFrame(results)

    print()
    print("=== Phase 3: Robustness Comparison ===")
    print(comparison.round(5).to_string(index=False))

    comparison.to_csv(ASSETS / "robustness_comparison.csv", index=False)

    # Step 2. Bootstrap confidence intervals for Model B
    print()
    print("=== Bootstrap Confidence Intervals ===")

    base = tracks.loc[tracks["n_plays"] >= 5].dropna(
        subset=list(PREDICTORS) + ["skip_2_rate"]
    )
    x_df = design(base, PREDICTORS)
    X = x_df.to_numpy()
    y = base["skip_2_rate"].to_numpy()
    weights = base["n_plays"].to_numpy()

    rng = np.random.default_rng(139)
    n_boot = 300
    estimates = []

    # Resample tracks with replacement
    for _ in range(n_boot):
        idx = rng.integers(0, len(base), len(base))
        w = np.sqrt(weights[idx])
        X_boot = X[idx] * w[:, None]
        y_boot = y[idx] * w
        beta = np.linalg.lstsq(X_boot, y_boot, rcond=None)[0]
        estimates.append(beta)

    estimates = np.asarray(estimates)
    original_fit = fit_model(base, "skip_2_rate")

    bootstrap = pd.DataFrame(
        {
            "feature": x_df.columns,
            "coef": original_fit.params.to_numpy(),
            "ci_lower": np.percentile(estimates, 2.5, axis=0),
            "ci_upper": np.percentile(estimates, 97.5, axis=0),
        }
    )

    bootstrap = bootstrap[bootstrap["feature"].isin(AUDIO)]
    print(bootstrap.round(4).to_string(index=False))

    bootstrap.to_csv(ASSETS / "robustness_bootstrap.csv", index=False)

    print()
    print("Phase 3 completed!")


if __name__ == "__main__":
    main()
