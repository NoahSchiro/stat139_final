from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.aggregate import AUDIO_FEATURES, aggregate
from src.eda import load_analysis_table
from src.model import design, fit_wls
from src.model_analysis import CONTROLS
from src.model_robustness import AUDIO as MODEL_B_AUDIO

ASSETS = Path("assets")


def run_anova(df):
    print()
    print("=== One-Way ANOVA by Listening Context ===")

    # One observation per session-context combination
    sessions = (df.groupby(["session_id", "context_type"], as_index=False).agg(
            skip_rate=("skip_2", "mean"),
            n_plays=("skip_2", "size"),
        )
    )

    # Match the EDA's minimum of 3 plays per group
    sessions = sessions[sessions["n_plays"] >= 3].copy()
    summary = sessions.groupby("context_type")["skip_rate"].agg(["size", "mean", "std"])
    print(summary.round(4).to_string())

    # Standard one-way ANOVA
    groups = [group["skip_rate"].to_numpy() for _, group in sessions.groupby("context_type")]
    f_stat, p_value = stats.f_oneway(*groups)

    print()
    print("=== ANOVA F-Test ===")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.6g}")

    # Permutation test (shuffle context labels)
    rng = np.random.default_rng(139)
    n_perm = 2000

    y = sessions["skip_rate"].to_numpy()
    labels = sessions["context_type"].astype("category").cat.codes.to_numpy()

    n = np.bincount(labels)
    k = len(n)
    total = len(y)
    grand_mean = y.mean()
    ss_total = np.sum((y - grand_mean) ** 2)

    extreme = 0

    for _ in range(n_perm):
        shuffled = rng.permutation(labels)
        sums = np.bincount(shuffled, weights=y, minlength=k)
        means = sums / n

        ss_between = np.sum(n * (means - grand_mean) ** 2)
        ss_within = ss_total - ss_between

        f_perm = (ss_between / (k - 1)) / (ss_within / (total - k))
        if f_perm >= f_stat: extreme += 1

    perm_p = (extreme + 1) / (n_perm + 1)

    print()
    print("=== Permutation Test ===")
    print(f"Permutations: {n_perm}")
    print(f"Permutation p-value: {perm_p:.6g}")

    # Additional check: account for shared sessions
    clustered = smf.ols("skip_rate ~ C(context_type)", data=sessions).fit(
        cov_type="cluster",
        cov_kwds={"groups": sessions["session_id"]}
    )

    indices = [i for i, name in enumerate(clustered.params.index) if name.startswith("C(context_type)")]
    R = np.eye(len(clustered.params))[indices]
    test = clustered.wald_test(R, scalar=True)

    cluster_p = float(test.pvalue)

    print()
    print("=== Session-Clustered Context Test ===")
    print(f"Wald p-value: {cluster_p:.6g}")

    summary.to_csv(ASSETS / "anova_context_summary.csv")

    pd.DataFrame([{
        "observations": len(sessions),
        "sessions": sessions["session_id"].nunique(),
        "F": f_stat,
        "anova_p": p_value,
        "permutation_p": perm_p,
        "session_clustered_p": cluster_p,
    }]).to_csv(ASSETS / "anova_tests.csv", index=False)


def run_group_tests(df):
    print()
    print("=== Grouped Nested F-Tests ===")

    tracks = aggregate(df).dropna(subset=list(CONTROLS + AUDIO_FEATURES))

    # Groups from the proposal and related audio features
    # AI Helped me organize this variable lol
    specifications = {
        "Original baseline": {
            "audio": AUDIO_FEATURES,
            "groups": {
                "mood": ("valence", "energy", "danceability"),
                "rhythm": (
                    "beat_strength", "bounciness",
                    "dyn_range_mean"
                ),
                "timing": ("tempo", "duration"),
            },
        },
        "Model B": {
            "audio": MODEL_B_AUDIO,
            "groups": {
                "mood": ("valence", "energy"),
                "timing": ("tempo", "duration"),
            },
        },
    }

    results = []
    for model_name, specification in specifications.items():
        audio = specification["audio"]
        columns = CONTROLS + audio
        full = fit_wls(tracks, columns) ## ful WLS model

        robust = sm.WLS( # HC3 version for robust joint tests
            tracks["skip_rate"],
            design(tracks, columns),
            weights=tracks["n_plays"]
        ).fit(cov_type="HC3")

        for group_name, features in specification["groups"].items():
            # Remove only the features being tested
            reduced_columns = tuple(col for col in columns if col not in features)
            reduced = fit_wls(tracks, reduced_columns)
            f_stat, p_value, df_diff = full.compare_f_test(reduced) ## classical nested F-test
            indices = [robust.params.index.get_loc(col) for col in features] # HC3-robust joint Wald test

            R = np.eye(len(robust.params))[indices]
            test = robust.wald_test(R, scalar=True)

            results.append({
                "model": model_name,
                "group": group_name,
                "features": ", ".join(features),
                "df": int(df_diff),
                "F": f_stat,
                "p": p_value,
                "HC3_p": float(test.pvalue),
                "delta_R2": full.rsquared - reduced.rsquared,
            })

    results = pd.DataFrame(results)

    # Correct robust p-values within each model's group tests
    results["HC3_p_fdr"] = results.groupby("model")["HC3_p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
    print(results.round(5).to_string(index=False))

    results.to_csv(ASSETS / "grouped_f_tests.csv", index=False)
    print()
    print("Saved grouped F-test results.")


def main():
    ASSETS.mkdir(exist_ok=True)
    df = load_analysis_table(Path("data/clean/session_tracks.csv"))
    run_anova(df)
    run_group_tests(df)
    print()
    print("Proposal hypothesis tests completed!")

if __name__ == "__main__": main()
