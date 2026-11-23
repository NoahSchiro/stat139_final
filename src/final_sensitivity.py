from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

from src.aggregate import aggregate
from src.eda import load_analysis_table
from src.model import design
from src.model_analysis import CONTROLS
from src.model_robustness import AUDIO

# Load the cleaned dataset
df = load_analysis_table(Path("data/clean/session_tracks.csv"))

# Identify plays without any outcome label
flags = ["skip_1", "skip_2", "skip_3", "not_skipped"]
unlabeled = ~df[flags].any(axis=1)
print("Unlabeled plays:", unlabeled.sum())

# Aggregate with and without unlabeled plays
original = aggregate(df)
filtered = aggregate(df.loc[~unlabeled].copy())

# Keep the same tracks for a fair comparison
common_ids = set(filtered["track_id_clean"])
original = original[original["track_id_clean"].isin(common_ids)]
filtered = filtered[filtered["track_id_clean"].isin(common_ids)]

predictors = tuple(CONTROLS) + tuple(AUDIO)


def fit_model(data: pd.DataFrame) -> RegressionResultsWrapper:
    X = design(data, predictors)
    return sm.WLS(data["skip_rate"], X, weights=data["n_plays"]).fit(cov_type="HC3")


model_original = fit_model(original)
model_filtered = fit_model(filtered)

print()
print("=== Unlabeled Play Sensitivity ===")
print("Tracks compared:", len(filtered))
print("Original R^2:", round(model_original.rsquared, 4))
print("Filtered R^2:", round(model_filtered.rsquared, 4))

for feature in ["duration", "speechiness", "loudness", "bounciness"]:
    print()
    print(f"{feature}")
    print("Original coefficient:", round(model_original.params[feature], 5))
    print("Filtered coefficient:", round(model_filtered.params[feature], 5))
    print("Original HC3 p-value:", round(model_original.pvalues[feature], 5))
    print("Filtered HC3 p-value:", round(model_filtered.pvalues[feature], 5))
