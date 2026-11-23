from pathlib import Path

import pandas as pd

from src.aggregate import AUDIO_FEATURES, aggregate
from src.eda import load_analysis_table
from src.model import design, fit_wls, vif_table
from src.model_analysis import CONTROLS

ASSETS = Path("assets")

RHYTHM = (
    "danceability",
    "bounciness",
    "beat_strength",
    "dyn_range_mean",
)

MODELS = {
    "A - Danceability": "danceability",
    "B - Bounciness": "bounciness",
    "C - Beat strength": "beat_strength",
    "D - Dynamic range": "dyn_range_mean",
}


def main() -> None:
    ASSETS.mkdir(exist_ok=True)

    df = load_analysis_table(Path("data/clean/session_tracks.csv"))
    tracks = aggregate(df).dropna(subset=list(CONTROLS + AUDIO_FEATURES))

    controls_only = fit_wls(tracks, CONTROLS)
    original = fit_wls(tracks, CONTROLS + AUDIO_FEATURES)

    results = []

    for name, selected in MODELS.items():
        audio = tuple(
            f
            for f in AUDIO_FEATURES
            if f not in {"mechanism", "organism"} and (f not in RHYTHM or f == selected)
        )

        fit = fit_wls(tracks, CONTROLS + audio)
        vif = vif_table(
            design(tracks, CONTROLS + audio), ASSETS / f"vif_{selected}.csv"
        )

        f_stat, p_value, _ = fit.compare_f_test(controls_only)
        _, p_drop, _ = original.compare_f_test(fit)

        results.append(
            {
                "Model": name,
                "R2": fit.rsquared,
                "Adj_R2": fit.rsquared_adj,
                "AIC": fit.aic,
                "Max_VIF": vif.max(),
                "Audio_F": f_stat,
                "Audio_p": p_value,
                "Removed_Features_p": p_drop,
                "Rhythm_Coef": fit.params[selected],
                "Rhythm_p": fit.pvalues[selected],
            }
        )

    comparison = pd.DataFrame(results).set_index("Model")

    print()
    print("=== ORIGINAL BASELINE ===")
    print(f"R^2: {original.rsquared:.4f}")
    print(f"Adjusted R^2: {original.rsquared_adj:.4f}")
    print(f"AIC: {original.aic:.2f}")

    print()
    print("=== REDUCED MODEL COMPARISON ===")
    print(comparison.round(4).to_string())

    comparison.to_csv(ASSETS / "model_comparison.csv")
    print()
    print("Saved assets/model_comparison.csv")


if __name__ == "__main__":
    main()
