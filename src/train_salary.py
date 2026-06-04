"""
train_salary.py — Regression: predict salary (LPA) for placed students
Software Career Predictor
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utils import SALARY_MODEL_PATH, print_section, print_subsection

MODELS = {
    "Linear Regression":     LinearRegression(),
    "Ridge Regression":      Ridge(alpha=1.0),
    "Random Forest":         RandomForestRegressor(n_estimators=200, max_depth=12,
                                                    random_state=42, n_jobs=-1),
    "Gradient Boosting":     GradientBoostingRegressor(n_estimators=200,
                                                        learning_rate=0.1,
                                                        max_depth=5,
                                                        random_state=42),
}


def evaluate_regressor(name, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "name": name,
        "r2":   r2_score(y_test, y_pred),
        "mae":  mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mse":  mean_squared_error(y_test, y_pred),
    }


def train_salary_models(X_train, X_test, y_train, y_test,
                         feature_names: list, plots_dir=None):
    """
    Train all regressors, evaluate, pick best by R², save it.
    Returns (best_model, results_list).
    """
    print_section("MODEL 3 — SALARY PREDICTION (Regression)")

    results = []
    trained = {}

    for name, model in MODELS.items():
        print(f"\n  Training {name} …", end=" ", flush=True)
        model.fit(X_train, y_train)
        metrics = evaluate_regressor(name, model, X_test, y_test)
        results.append(metrics)
        trained[name] = model
        print(f"R²={metrics['r2']:.4f}  MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print_subsection("Comparison table")
    header = f"  {'Model':<22} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'MSE':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        print(f"  {r['name']:<22} {r['r2']:>8.4f} {r['mae']:>8.2f} "
              f"{r['rmse']:>8.2f} {r['mse']:>12.2f}")

    best       = max(results, key=lambda x: x["r2"])
    best_model = trained[best["name"]]
    print(f"\n  ✓ Best model: {best['name']}  (R² = {best['r2']:.4f})")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if plots_dir is not None:
        _plot_actual_vs_predicted(best_model, X_test, y_test,
                                   best["name"], plots_dir)
        _plot_residuals(best_model, X_test, y_test,
                        best["name"], plots_dir)
        _plot_feature_importance(best_model, best["name"],
                                  feature_names, plots_dir)
        _plot_metrics_comparison(results, plots_dir)

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(best_model, SALARY_MODEL_PATH)
    print(f"\n  Saved → {SALARY_MODEL_PATH}")

    return best_model, results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
def _plot_actual_vs_predicted(model, X_test, y_test, name, plots_dir):
    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.35, edgecolors="none",
               color="#4C72B0", s=18, label="Predictions")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Salary (LPA)")
    ax.set_ylabel("Predicted Salary (LPA)")
    ax.set_title(f"Salary — Actual vs Predicted\n({name})", fontsize=13)
    ax.legend()
    r2 = r2_score(y_test, y_pred)
    ax.text(0.05, 0.93, f"R² = {r2:.4f}", transform=ax.transAxes,
            fontsize=11, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    path = plots_dir / "salary_actual_vs_predicted.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path.name}")


def _plot_residuals(model, X_test, y_test, name, plots_dir):
    y_pred    = model.predict(X_test)
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.35, edgecolors="none",
                    color="#DD8452", s=18)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predicted Salary (LPA)")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")

    # Distribution of residuals
    axes[1].hist(residuals, bins=50, color="#55A868", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    fig.suptitle(f"Salary — Residual Analysis ({name})", fontsize=13)
    plt.tight_layout()
    path = plots_dir / "salary_residuals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path.name}")


def _plot_feature_importance(model, name, feature_names, plots_dir):
    if not hasattr(model, "feature_importances_"):
        # For linear models, use absolute coefficients
        if hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            return
    else:
        importances = model.feature_importances_

    idx = np.argsort(importances)[-10:][::-1]
    top_features = [feature_names[i] for i in idx]
    top_vals     = importances[idx]
    # Normalise to [0,1] for display
    top_vals_norm = top_vals / top_vals.sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = plt.cm.viridis(np.linspace(0.2, 0.85, len(top_features)))
    bars = ax.barh(top_features[::-1], top_vals_norm[::-1], color=palette[::-1])
    ax.set_title(f"Salary — Top 10 Feature Importances\n({name})", fontsize=13)
    ax.set_xlabel("Normalised Importance")
    for bar, val in zip(bars, top_vals_norm[::-1]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    path = plots_dir / "salary_feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path.name}")


def _plot_metrics_comparison(results, plots_dir):
    names = [r["name"] for r in results]
    r2s   = [r["r2"]   for r in results]
    maes  = [r["mae"]  for r in results]
    rmses = [r["rmse"] for r in results]

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    palette = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]

    for ax, vals, title, ylabel in zip(
        axes,
        [r2s, maes, rmses],
        ["R² Score (higher=better)", "MAE — Mean Absolute Error (lower=better)",
         "RMSE (lower=better)"],
        ["R²", "LPA", "LPA"],
    ):
        bars = ax.bar(x, vals, color=palette[:len(names)], edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)

    fig.suptitle("Salary Models — Metrics Comparison", fontsize=13)
    plt.tight_layout()
    path = plots_dir / "salary_metrics_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path.name}")
