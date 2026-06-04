"""
train_placement.py — Binary classification: will a student be placed?
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

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay,
)

from utils import PLACEMENT_MODEL_PATH, print_section, print_subsection

# ─────────────────────────────────────────────────────────────────────────────
# Model zoo
# ─────────────────────────────────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=12,
                                                   random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200,
                                                       learning_rate=0.1,
                                                       max_depth=5,
                                                       random_state=42),
}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate one model
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_classifier(name, model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "name":      name,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Train & compare
# ─────────────────────────────────────────────────────────────────────────────
def train_placement_models(X_train, X_test, y_train, y_test,
                            feature_names: list, plots_dir=None):
    """
    Train all classifiers, evaluate, pick best by F1, save it.
    Returns (best_model, results_list).
    """
    print_section("MODEL 1 — PLACEMENT PREDICTION (Binary Classification)")

    results = []
    trained = {}

    for name, model in MODELS.items():
        print(f"\n  Training {name} …", end=" ", flush=True)
        model.fit(X_train, y_train)
        metrics = evaluate_classifier(name, model, X_test, y_test)
        results.append(metrics)
        trained[name] = model
        print(f"Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
              f"AUC={metrics['roc_auc']:.4f}")

    # ── Print full comparison table ───────────────────────────────────────────
    print_subsection("Comparison table")
    header = f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        print(f"  {r['name']:<22} {r['accuracy']:>9.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['roc_auc']:>9.4f}")

    # ── Select best model ─────────────────────────────────────────────────────
    best = max(results, key=lambda x: x["f1"])
    best_model = trained[best["name"]]
    print(f"\n  ✓ Best model: {best['name']}  (F1 = {best['f1']:.4f})")

    # ── Full classification report for best model ─────────────────────────────
    print_subsection(f"Classification report — {best['name']}")
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best,
                                  target_names=["Not Placed", "Placed"]))

    # ── Plots ─────────────────────────────────────────────────────────────────
    if plots_dir is not None:
        _plot_confusion_matrix(best_model, X_test, y_test,
                                best["name"], plots_dir)
        _plot_roc_curves(trained, X_test, y_test, plots_dir)
        _plot_feature_importance(best_model, best["name"],
                                  feature_names, plots_dir)
        _plot_metrics_comparison(results, plots_dir)

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(best_model, PLACEMENT_MODEL_PATH)
    print(f"\n  Saved → {PLACEMENT_MODEL_PATH}")

    return best_model, results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
def _plot_confusion_matrix(model, X_test, y_test, name, plots_dir):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Placed", "Placed"],
                yticklabels=["Not Placed", "Placed"], ax=ax)
    ax.set_title(f"Placement — Confusion Matrix\n({name})", fontsize=13)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    path = plots_dir / "placement_confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path.name}")


def _plot_roc_curves(trained, X_test, y_test, plots_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in trained.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test,
                                        name=name, ax=ax)
    ax.set_title("Placement — ROC Curves (all models)", fontsize=13)
    plt.tight_layout()
    path = plots_dir / "placement_roc_curves.png"
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
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))
    bars = ax.barh(top_features[::-1], top_vals[::-1], color=colors[::-1])
    ax.set_title(f"Placement — Top 10 Feature Importances\n({name})", fontsize=13)
    ax.set_xlabel("Importance")
    for bar, val in zip(bars, top_vals[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    path = plots_dir / "placement_feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path.name}")


def _plot_metrics_comparison(results, plots_dir):
    metrics_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_names  = [r["name"] for r in results]
    x = np.arange(len(model_names))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    for i, key in enumerate(metrics_keys):
        vals = [r[key] for r in results]
        ax.bar(x + i * width, vals, width, label=key.upper(), color=colors[i])

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Placement Models — Metrics Comparison", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylabel("Score")
    plt.tight_layout()
    path = plots_dir / "placement_metrics_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path.name}")
