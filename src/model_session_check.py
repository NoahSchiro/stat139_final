from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from src.model_interactions import AUDIO, CONTEXT, INTERACTIONS

ASSETS = Path("assets")


def main():
    ASSETS.mkdir(exist_ok=True)
    df = pd.read_csv("data/clean/session_tracks.csv")

    # Keep tracks with at least 5 total plays
    counts = df.groupby("track_id_clean")["skip_2"].transform("size")
    df = df.loc[counts >= 5].copy()

    # Convert Boolean columns to integers
    for col in ("skip_1", "skip_2", "skip_3", "hist_user_behavior_is_shuffle", "premium"):
        df[col] = df[col].astype(int)

    # Prepare track-context predictors
    track_cols = AUDIO + ("us_popularity_estimate", "release_year")

    grouped = (
        df.groupby(["track_id_clean", "context_type"], as_index=False)
        .agg(
            prop_shuffle=("hist_user_behavior_is_shuffle", "mean"),
            prop_premium=("premium", "mean"),
            mean_position=("session_position", "mean"),
            **{col: (col, "first") for col in track_cols}
        )
        .dropna()
        .copy()
    )

    # Standardize using unique tracks
    unique = grouped.drop_duplicates("track_id_clean")

    for col in track_cols:
        grouped[col] = (grouped[col] - unique[col].mean()) / unique[col].std()

    # Merge track-context predictors back to individual plays
    plays = df[
        [
            "track_id_clean",
            "context_type",
            "session_id",
            "skip_1",
            "skip_2",
            "skip_3",
        ]
    ].merge(
        grouped,
        on=["track_id_clean", "context_type"],
        how="inner",
        validate="m:1"
    )

    # Build interaction model
    terms = list(AUDIO) + [
        "us_popularity_estimate",
        "release_year",
        "prop_shuffle",
        "prop_premium",
        "mean_position",
        CONTEXT,
    ]

    terms += [f"{feature}:{CONTEXT}" for feature in INTERACTIONS]
    predictors = " + ".join(terms)

    # Track and session cluster identifiers
    track_codes = pd.factorize(plays["track_id_clean"])[0]
    session_codes = pd.factorize(plays["session_id"])[0]
    groups = np.column_stack([track_codes, session_codes])

    print()
    print("=== Session Dependence Check ===")
    print("Plays:", len(plays))
    print("Tracks:", plays["track_id_clean"].nunique())
    print("Sessions:", plays["session_id"].nunique())

    # Test interactions for all three skip definitions
    print()
    print("=== Alternative Skip Definition Checks ===")

    all_results = []
    for outcome in ("skip_1", "skip_2", "skip_3"):
        formula = f"{outcome} ~ {predictors}"
        model = smf.ols(formula, data=plays).fit()

        # Standard errors clustered by track
        track_fit = model.get_robustcov_results(cov_type="cluster", groups=track_codes)

        # Standard errors clustered by track and session
        both_fit = model.get_robustcov_results(cov_type="cluster", groups=groups)
        names = model.model.exog_names

        # Joint test of all interaction terms
        indices = [i for i, name in enumerate(names) if f":{CONTEXT}" in name]
        R = np.eye(len(names))[indices]
        joint_track = float(track_fit.wald_test(R, scalar=True).pvalue)
        joint_both = float(both_fit.wald_test(R, scalar=True).pvalue)

        print()
        print(f"=== {outcome} ===")
        print(f"R^2: {model.rsquared:.4f}")
        print(f"Track-clustered joint p: {joint_track:.6g}")
        print(f"Two-way clustered joint p: {joint_both:.6g}")

        # Feature-level Wald tests
        feature_results = []
        for feature in INTERACTIONS:
            indices = [i for i, name in enumerate(names) if name.startswith(f"{feature}:")]
            R = np.eye(len(names))[indices]
            p_track = float(track_fit.wald_test(R, scalar=True).pvalue)
            p_both = float(both_fit.wald_test(R, scalar=True).pvalue)

            feature_results.append({
                "outcome": outcome,
                "feature": feature,
                "R2": model.rsquared,
                "joint_track_p": joint_track,
                "joint_two_way_p": joint_both,
                "track_p": p_track,
                "two_way_p": p_both,
            })

        # FDR correction across the four feature tests
        adjusted = multipletests([row["two_way_p"] for row in feature_results], method="fdr_bh")[1]
        print()
        print("=== Feature-Level Tests ===")

        for row, p_fdr in zip(feature_results, adjusted):
            row["two_way_p_fdr"] = p_fdr
            print(
                f"{row['feature']}: "
                f"track p={row['track_p']:.5g}, "
                f"two-way p={row['two_way_p']:.5g}, "
                f"FDR p={p_fdr:.5g}"
            )
            all_results.append(row)

    # Save results
    results = pd.DataFrame(all_results)

    results.to_csv(ASSETS / "session_interaction_robustness.csv", index=False)

    print()
    print("Saved session_interaction_robustness.csv")
    print()
    print("Session dependence and robustness checks completed!")


if __name__ == "__main__": main()
