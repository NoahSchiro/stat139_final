from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats
from src.model import design
from src.model_robustness import prepare_data, PREDICTORS

ASSETS = Path("assets")


def main():
    ASSETS.mkdir(exist_ok=True)

    # Load and prepare Model B (selected model) data
    tracks = prepare_data()
    tracks = tracks.loc[tracks["n_plays"] >= 5].dropna(subset=list(PREDICTORS) + ["skip_2_rate"])

    x = design(tracks, PREDICTORS)
    y = tracks["skip_2_rate"]
    weights = tracks["n_plays"]

    # Fit selected Model B with robust standard errors
    fit = sm.WLS(y, x, weights=weights).fit(cov_type="HC3")

    fitted = fit.fittedvalues
    residuals = (y - fitted) * np.sqrt(weights)

    # Calculate Cook's distance using equivalent weighted OLS
    sqrt_w = np.sqrt(weights.to_numpy())
    weighted_x = x.to_numpy() * sqrt_w[:, None]
    weighted_y = y.to_numpy() * sqrt_w

    weighted_fit = sm.OLS(weighted_y, weighted_x).fit()
    cooks = weighted_fit.get_influence().cooks_distance[0]

    # Step 1. Look at Model diagnostics
    print("")
    print("=== Final Model Diagnostics ===")
    print(f"Tracks: {len(tracks)}")
    print(f"R^2: {fit.rsquared:.4f}")
    print(f"Adjusted R^2: {fit.rsquared_adj:.4f}")
    print(f"Minimum prediction: {fitted.min():.4f}")
    print(f"Maximum prediction: {fitted.max():.4f}")

    outside = ((fitted < 0) | (fitted > 1)).sum()
    print(f"Predictions outside [0, 1]: {outside}")

    cutoff = 4 / len(tracks)
    print(f"Maximum Cook's distance: {cooks.max():.6f}")
    print(f"Tracks with Cook's D > 4/n: {(cooks > cutoff).sum()}")

    # Step 2. Examine the five most influential tracks
    top = np.argsort(cooks)[-5:][::-1]

    print("")
    print("=== Top 5 Influential Tracks ===")

    for i in top:
        print(
            f"Track: {tracks.index[i]}, "
            f"Plays: {weights.iloc[i]}, "
            f"Skip rate: {y.iloc[i]:.4f}, "
            f"Cook's D: {cooks[i]:.4f}"
        )

    # Step 3. Refit without the most influential track
    mask = np.ones(len(tracks), dtype=bool)
    mask[top[0]] = False

    refit = sm.WLS(
        y.iloc[mask],
        x.iloc[mask],
        weights=weights.iloc[mask]
    ).fit(cov_type="HC3")

    print("")
    print("=== Influence Sensitivity ===")
    print(f"Original R^2: {fit.rsquared:.4f}")
    print(f"Without top track R^2: {refit.rsquared:.4f}")

    for feature in (
        "bounciness",
        "duration",
        "speechiness",
        "loudness"
    ):
        print(
            f"{feature}: "
            f"original={fit.params[feature]:.4f}, "
            f"without top track={refit.params[feature]:.4f}"
        )

    # Step 4. Diagnostic visualizations
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Residuals vs fitted
    axes[0].scatter(fitted, residuals, s=5, alpha=0.3)
    axes[0].axhline(0, color="gray", linestyle="--")
    axes[0].set_title("Residuals vs Fitted")
    axes[0].set_xlabel("Predicted skip rate")
    axes[0].set_ylabel("Weighted residual")

    # Normal QQ plot
    stats.probplot(residuals, plot=axes[1])
    axes[1].set_title("Normal QQ Plot")

    # Cook's distance
    axes[2].scatter(range(len(cooks)), cooks, s=5)
    axes[2].axhline(cutoff, color="red", linestyle="--")
    axes[2].set_title("Cook's Distance")
    axes[2].set_xlabel("Track index")
    axes[2].set_ylabel("Cook's distance")

    fig.tight_layout()
    fig.savefig(ASSETS / "final_model_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("")
    print("Saved final_model_diagnostics.png")


if __name__ == "__main__": main()
