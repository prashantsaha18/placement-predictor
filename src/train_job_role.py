"""
train_job_role.py — Multi-class classification: predict job role for placed students
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

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)

from utils import JOB_ROLE_MODEL_PATH, print_section, print_subsection

MODELS = {
    "Decision Tree":     DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest":     RandomForestClassifier(n_estimators=200, max_depth=12,
                                                 random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200,
                                                     learning_rate=0.1,
                                                     max_depth=5,
                                                     random_state=42),
}


def train_job_role_models(X_train, X_test, y_train, y_test,
                           feature_names: list, le_role,
                           plots_dir=None):
    """
    Train all classifiers for job-role prediction, evaluate, pick best by accuracy.
    Returns (best_model, results_list).
    """
    print_section("MODEL 2 — JOB ROLE PREDICTION (Multi-Class Classification)")

    class_names = list(le_role.classes_)
    results  = []
    trained  = {}

    for name, model in MODELS.items():
        print(f"\n  Training {name} …", end=" ", flush=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({"name": name, "accuracy": acc})
        trained[name] = model
        print(f"Accuracy = {acc:.4f}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print_subsection("Comparison table")
    print(f"  {'Model':<22} {'Accuracy':>10}")
    print("  " + "-" * 34)
    for r in results:
        print(f"  {r['name']:<22} {r['accuracy']:>10.4f}")

    best     = max(results, key=lambda x: x["accuracy"])
    best_model = trained[best["name"]]
    print(f"\n  ✓ Best model: {best['name']}  (Accuracy = {best['accuracy']:.4f})")

    # ── Full classification report ────────────────────────────────────────────
    print_subsection(f"Classification report — {best['name']}")
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best, target_names=class_names))

    # ── Plots ─────────────────────────────────────────────────────────────────
    if plots_dir is not None:
        _plot_confusion_matrix(best_model, X_test, y_test,
                                best["name"], class_names, plots_dir)
        _plot_feature_importance(best_model, best["name"],
                                  feature_names, plots_dir)
        _plot_accuracy_comparison(results, plots_dir)

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(best_model, JOB_ROLE_MODEL_PATH)
    print(f"\n  Saved → {JOB_ROLE_MODEL_PATH}")

    return best_model, results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
def _plot_confusion_matrix(model, X_test, y_test, name, class_names, plots_dir):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"Job Role — Confusion Matrix (counts)\n{name}", fontsize=12)
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")

    # Percentages
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f"Job Role — Confusion Matrix (%)\n{name}", fontsize=12)
    axes[1].set_ylabel("Actual")
    axes[1].set_xlabel("Predicted")

    plt.tight_layout()
    path = plots_dir / "job_role_confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path.name}")


def _plot_feature_importance(model, name, feature_names, plots_dir):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[-10:][::-1]
    top_features = [feature_names[i] for i in idx]
    top_vals     = importances[idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(top_features)))
    bars = ax.barh(top_features[::-1], top_vals[::-1], color=palette[::-1])
    ax.set_title(f"Job Role — Top 10 Feature Importances\n({name})", fontsize=13)
    ax.set_xlabel("Importance")
    for bar, val in zip(bars, top_vals[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    path = plots_dir / "job_role_feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path.name}")


def _plot_accuracy_comparison(results, plots_dir):
    names = [r["name"] for r in results]
    accs  = [r["accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4C72B0", "#55A868", "#DD8452"]
    bars = ax.bar(names, accs, color=colors[:len(names)], edgecolor="white", width=0.5)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_title("Job Role Models — Accuracy Comparison", fontsize=13)
    ax.set_ylabel("Accuracy")
    plt.tight_layout()
    path = plots_dir / "job_role_accuracy_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path.name}")
