from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ASSETS = Path("assets")

# Model B (Selected Model): 11 retained audio features
AUDIO = (
    "acousticness", "bounciness", "energy", "flatness",
    "instrumentalness", "liveness", "loudness",
    "speechiness", "tempo", "valence", "duration"
)

# Features tested for interactions
INTERACTIONS = ("bounciness", "energy", "duration", "speechiness")
CONTEXT = "C(context_type, Treatment(reference='user_collection'))"


def main():
    ASSETS.mkdir(exist_ok=True)
    df = pd.read_csv("data/clean/session_tracks.csv")

    # Keep tracks with at least 5 total plays
    counts = df.groupby("track_id_clean")["skip_2"].transform("size")
    df = df.loc[counts >= 5].copy()

    for col in ("skip_2", "hist_user_behavior_is_shuffle", "premium"):
        df[col] = df[col].astype(int)

    # Aggregate by track and listening context
    track_cols = AUDIO + ("us_popularity_estimate", "release_year")

    agg_dict = {
        "n_plays": ("skip_2", "size"),
        "skip_rate": ("skip_2", "mean"),
        "prop_shuffle": ("hist_user_behavior_is_shuffle", "mean"),
        "prop_premium": ("premium", "mean"),
        "mean_position": ("session_position", "mean"),
    }

    agg_dict.update({col: (col, "first") for col in track_cols})

    grouped = (
        df.groupby(["track_id_clean", "context_type"], as_index=False)
        .agg(**agg_dict)
        .dropna()
        .copy()
    )

    # Standardize using unique tracks
    unique_tracks = grouped.drop_duplicates("track_id_clean")

    for col in track_cols:
        mean = unique_tracks[col].mean()
        sd = unique_tracks[col].std()
        grouped[col] = (grouped[col] - mean) / sd

    # Main-effects and interaction formulas
    terms = list(AUDIO) + [
        "us_popularity_estimate",
        "release_year",
        "prop_shuffle",
        "prop_premium",
        "mean_position",
        CONTEXT,
    ]

    base_formula = "skip_rate ~ " + " + ".join(terms)

    interaction_terms = [f"{feature}:{CONTEXT}" for feature in INTERACTIONS]
    full_formula = base_formula + " + " + " + ".join(interaction_terms)

    # Weighted regression with track-clustered standard errors
    def fit(formula, data):
        return smf.wls(formula, data=data, weights=data["n_plays"]).fit(
            cov_type="cluster",
            cov_kwds={"groups": data["track_id_clean"]}
        )

    baseline = fit(base_formula, grouped)
    interaction = fit(full_formula, grouped)

    # Step 1. Overall model comparison
    print()
    print("=== Phase 2: Model Comparison ===")
    print("Track-context observations:", len(grouped))
    print("Unique tracks:", grouped["track_id_clean"].nunique())
    print("Plays:", grouped["n_plays"].sum())
    print()
    print(f"Main-effects R^2: {baseline.rsquared:.4f}")
    print(f"Interaction R^2: {interaction.rsquared:.4f}")
    print(f"Main-effects AIC: {baseline.aic:.2f}")
    print(f"Interaction AIC: {interaction.aic:.2f}")

    # Joint Wald test for all interactions
    names = [name for name in interaction.params.index if f":{CONTEXT}" in name]
    indices = [interaction.params.index.get_loc(name) for name in names]

    R = np.eye(len(interaction.params))[indices]
    test = interaction.wald_test(R, scalar=True)

    print()
    print("=== Joint Interaction Test ===")
    print(f"Chi-square: {float(test.statistic):.4f}")
    print(f"Degrees of freedom: {int(test.df_denom)}")
    print(f"p-value: {float(test.pvalue):.4e}")

    # Step 2. Individual interaction coefficients
    results = pd.DataFrame({
        "coef": interaction.params[names],
        "se": interaction.bse[names],
        "p": interaction.pvalues[names],
    })

    results["p_fdr"] = multipletests(results["p"], method="fdr_bh")[1]

    print()
    print("=== Individual Interaction Coefficients ===")
    print(results.round(5).to_string())
    results.to_csv(ASSETS / "interaction_coefficients.csv")

    # Step 3. Context-specific audio slopes
    slopes = []

    for feature in INTERACTIONS:
        for context in sorted(grouped["context_type"].unique()):
            contrast = np.zeros(len(interaction.params))
            contrast[interaction.params.index.get_loc(feature)] = 1

            if context != "user_collection":
                name = f"{feature}:{CONTEXT}[T.{context}]"
                contrast[interaction.params.index.get_loc(name)] = 1

            estimate = interaction.t_test(contrast)
            lo, hi = estimate.conf_int()[0]

            slopes.append({
                "feature": feature,
                "context": context,
                "coef": float(np.asarray(estimate.effect).item()),
                "lo": float(lo),
                "hi": float(hi),
                "p": float(np.asarray(estimate.pvalue).item()),
            })

    slopes = pd.DataFrame(slopes)

    print()
    print("=== Context-Specific Audio Slopes ===")
    print(
        slopes.pivot(
            index="context",
            columns="feature",
            values="coef"
        ).round(4).to_string()
    )

    slopes.to_csv(ASSETS / "context_audio_slopes.csv", index=False)

    # Step 4. Feature-level Wald tests
    print()
    print("=== Feature-Level Wald Tests ===")

    feature_tests = []

    for feature in INTERACTIONS:
        terms = [name for name in interaction.params.index if name.startswith(f"{feature}:")]
        indices = [interaction.params.index.get_loc(name) for name in terms]

        R = np.eye(len(interaction.params))[indices]
        test = interaction.wald_test(R, scalar=True)

        feature_tests.append({
            "feature": feature,
            "chi2": float(test.statistic),
            "df": len(terms),
            "p": float(test.pvalue),
        })

    feature_tests = pd.DataFrame(feature_tests)

    feature_tests["p_fdr"] = multipletests(feature_tests["p"], method="fdr_bh")[1]

    print(feature_tests.round(5).to_string(index=False))
    feature_tests.to_csv(ASSETS / "interaction_feature_tests.csv", index=False)

    # Step 5. Plot context-specific slopes with 95% CIs
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for ax, feature in zip(axes.flat, INTERACTIONS):
        subset = slopes[slopes["feature"] == feature].sort_values("coef")
        y = np.arange(len(subset))

        ax.errorbar(subset["coef"] * 100, y,
            xerr=[(subset["coef"] - subset["lo"]) * 100, (subset["hi"] - subset["coef"]) * 100],
            fmt="o",
            capsize=4
        )

        ax.set_yticks(y, subset["context"])
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_title(feature.capitalize())
        ax.set_xlabel("Skip rate change (percentage points)")

    fig.suptitle("Audio Feature Associations by Listening Context", fontsize=14)

    fig.tight_layout()
    fig.savefig(ASSETS / "interaction_slopes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print()
    print("Saved interaction_slopes.png")

    # Step 6. Minimum-play sensitivity analysis
    print()
    print("=== Minimum-Play Sensitivity ===")

    sensitivity = []

    for threshold in (1, 3, 5):
        subset = grouped[grouped["n_plays"] >= threshold].copy()
        base_fit = fit(base_formula, subset)
        full_fit = fit(full_formula, subset)
        terms = [name for name in full_fit.params.index if ":C(context_type" in name]
        indices = [full_fit.params.index.get_loc(name) for name in terms]
        R = np.eye(len(full_fit.params))[indices]
        test = full_fit.wald_test(R, scalar=True)

        sensitivity.append({
            "min_plays": threshold,
            "observations": len(subset),
            "tracks": subset["track_id_clean"].nunique(),
            "R2_main": base_fit.rsquared,
            "R2_interaction": full_fit.rsquared,
            "delta_R2": full_fit.rsquared - base_fit.rsquared,
            "chi2": float(test.statistic),
            "p": float(test.pvalue),
        })

    sensitivity = pd.DataFrame(sensitivity)
    print(sensitivity.round(5).to_string(index=False))
    sensitivity.to_csv(ASSETS / "interaction_sensitivity.csv", index=False)
    print()
    print("Phase 2 analyses completed!")


if __name__ == "__main__": main()
