from pathlib import Path

from src.aggregate import AUDIO_FEATURES, aggregate
from src.eda import load_analysis_table
from src.model import coefficient_table, design, fit_wls, vif_table

ASSETS = Path("assets")

# Same controls as the original baseline
CONTROLS = (
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

# Remove highly correlated or less interpretable features
REMOVE = {
    "beat_strength",
    "bounciness",
    "dyn_range_mean",
    "mechanism",
    "organism",
}

REDUCED_AUDIO = tuple(f for f in AUDIO_FEATURES if f not in REMOVE)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)

    # Load and aggregate the cleaned dataset
    df = load_analysis_table(Path("data/clean/session_tracks.csv"))
    tracks = aggregate(df).dropna(subset=list(CONTROLS + AUDIO_FEATURES))

    # Fit models on the same observations
    controls_only = fit_wls(tracks, CONTROLS)
    original = fit_wls(tracks, CONTROLS + AUDIO_FEATURES)
    reduced = fit_wls(tracks, CONTROLS + REDUCED_AUDIO)

    # Compare our model performance
    print()
    print("=== Model Comparison ===")
    print(f"Tracks: {len(tracks)}")
    print(f"Original R^2: {original.rsquared:.4f}")
    print(f"Reduced R^2: {reduced.rsquared:.4f}")
    print(f"Original adjusted R^2: {original.rsquared_adj:.4f}")
    print(f"Reduced adjusted R^2: {reduced.rsquared_adj:.4f}")
    print(f"Original AIC: {original.aic:.2f}")
    print(f"Reduced AIC: {reduced.aic:.2f}")

    # Test whether retained audio features add explanatory power
    f_stat, p_value, _ = reduced.compare_f_test(controls_only)
    print()
    print("=== Nested F-Test: Reduced Audio Block ===")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.4e}")

    # Now we test whether excluded features jointly add explanatory power
    f_drop, p_drop, _ = original.compare_f_test(reduced)
    print()
    print("=== F-Test: Excluded Audio Features ===")
    print(f"F-statistic: {f_drop:.4f}")
    print(f"p-value: {p_drop:.4e}")

    # Check for multicollinearity
    x = design(tracks, CONTROLS + REDUCED_AUDIO)
    vif = vif_table(x, ASSETS / "reduced_vif.csv")

    print()
    print("=== Reduced Model VIF ===")
    print(vif.round(2).to_string())

    # Save and display coefficients
    coefs = coefficient_table(reduced, ASSETS / "reduced_coefficients.csv")

    print()
    print("=== Reduced Audio Coefficients ===")
    print(coefs.loc[list(REDUCED_AUDIO), ["coef", "se", "p"]].round(4))
    print()
    print("=== Reduced Model Summary ===")
    print(reduced.summary())


if __name__ == "__main__":
    main()
