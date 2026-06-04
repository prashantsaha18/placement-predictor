"""
main.py — Master pipeline: runs EDA, preprocessing, all three models, and demo predictions
Software Career Predictor

Run from the project root:
    cd SoftwareCareerPredictor
    python src/main.py
"""

import sys
import os
import pathlib
import warnings
warnings.filterwarnings("ignore")

# ── Make src/ importable ──────────────────────────────────────────────────────
SRC_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

ROOT_DIR  = SRC_DIR.parent
PLOTS_DIR = ROOT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

from utils import print_section, print_subsection
from preprocess import run_preprocessing
from eda_plots import (
    load,
    plot_placement_distribution,
    plot_job_role_distribution,
    plot_salary_distribution,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_cgpa_vs_salary,
    plot_skill_radar,
)
from train_placement import train_placement_models
from train_job_role   import train_job_role_models
from train_salary     import train_salary_models


# ─────────────────────────────────────────────────────────────────────────────
def run_eda():
    print_section("STEP 1 — EXPLORATORY DATA ANALYSIS")
    df = load()
    print(f"  Generating EDA plots → {PLOTS_DIR}")
    plot_placement_distribution(df)
    plot_job_role_distribution(df)
    plot_salary_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_distributions(df)
    plot_cgpa_vs_salary(df)
    plot_skill_radar(df)
    print(f"  ✓  7 EDA plots saved")


# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline():
    print("\n" + "█" * 62)
    print("  SOFTWARE CAREER PREDICTOR — FULL ML PIPELINE")
    print("█" * 62)

    # ── EDA ───────────────────────────────────────────────────────────────────
    run_eda()

    # ── Pre-processing ────────────────────────────────────────────────────────
    print_section("STEP 2 — PREPROCESSING")
    splits, preprocessor, le_role, feature_names, df = run_preprocessing()

    X_tr_p, X_te_p, y_tr_p, y_te_p = splits["placement"]
    X_tr_r, X_te_r, y_tr_r, y_te_r = splits["job_role"]
    X_tr_s, X_te_s, y_tr_s, y_te_s = splits["salary"]

    # ── Model 1: Placement ────────────────────────────────────────────────────
    placement_model, p_results = train_placement_models(
        X_tr_p, X_te_p, y_tr_p, y_te_p,
        feature_names=feature_names,
        plots_dir=PLOTS_DIR,
    )

    # ── Model 2: Job Role ─────────────────────────────────────────────────────
    job_role_model, jr_results = train_job_role_models(
        X_tr_r, X_te_r, y_tr_r, y_te_r,
        feature_names=feature_names,
        le_role=le_role,
        plots_dir=PLOTS_DIR,
    )

    # ── Model 3: Salary ───────────────────────────────────────────────────────
    salary_model, sal_results = train_salary_models(
        X_tr_s, X_te_s, y_tr_s, y_te_s,
        feature_names=feature_names,
        plots_dir=PLOTS_DIR,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print_section("PIPELINE SUMMARY")

    best_p  = max(p_results,   key=lambda x: x["f1"])
    best_jr = max(jr_results,  key=lambda x: x["accuracy"])
    best_s  = max(sal_results, key=lambda x: x["r2"])

    print(f"\n  {'Model':<28} {'Metric':<12} {'Score':>8}")
    print(f"  {'-'*50}")
    print(f"  {'Placement  (' + best_p['name'] + ')':<28} {'F1':<12} {best_p['f1']:>8.4f}")
    print(f"  {'Job Role   (' + best_jr['name'] + ')':<28} {'Accuracy':<12} {best_jr['accuracy']:>8.4f}")
    print(f"  {'Salary     (' + best_s['name'] + ')':<28} {'R²':<12} {best_s['r2']:>8.4f}")

    from utils import MODELS_DIR
    print(f"\n  Saved artefacts in: {MODELS_DIR}")
    for f in sorted(MODELS_DIR.glob("*.pkl")):
        print(f"    • {f.name}")

    print(f"\n  Plots saved in: {PLOTS_DIR}")
    for f in sorted(PLOTS_DIR.glob("*.png")):
        print(f"    • {f.name}")

    # ── Live demo predictions ─────────────────────────────────────────────────
    print_section("STEP 4 — LIVE DEMO PREDICTIONS")
    _demo_predictions()

    print("\n" + "█" * 62)
    print("  PIPELINE COMPLETE")
    print("█" * 62 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
def _demo_predictions():
    """Run predict_all() on three sample students and pretty-print results."""
    from predict import predict_all

    samples = [
        dict(
            label="⭐ High-performer (CSE, Tier-1)",
            cgpa=9.1, python_skill=9, dsa_skill=9, ml_skill=8, web_dev_skill=7,
            coding_score=92, communication_score=88, aptitude_score=93,
            internships=3, projects=5, backlogs=0, resume_score=91, skill_score=88,
            branch="CSE", college_tier=1,
        ),
        dict(
            label="📊 Average student (ECE, Tier-2)",
            cgpa=7.2, python_skill=6, dsa_skill=6, ml_skill=5, web_dev_skill=7,
            coding_score=68, communication_score=70, aptitude_score=72,
            internships=1, projects=2, backlogs=1, resume_score=70, skill_score=64,
            branch="ECE", college_tier=2,
        ),
        dict(
            label="⚠️  Struggling (Mechanical, Tier-3)",
            cgpa=5.8, python_skill=4, dsa_skill=3, ml_skill=2, web_dev_skill=4,
            coding_score=48, communication_score=55, aptitude_score=50,
            internships=0, projects=1, backlogs=3, resume_score=52, skill_score=44,
            branch="Mechanical", college_tier=3,
        ),
    ]

    for s in samples:
        label = s.pop("label")
        result = predict_all(**s)
        p   = result["placement"]
        jr  = result["job_role"]
        sal = result["salary"]

        print(f"\n  {label}")
        status = "✓ PLACED" if p["placed"] else "✗ NOT PLACED"
        print(f"    Placement  : {status}  (confidence {p['probability']:.1%})")
        if p["placed"]:
            print(f"    Job Role   : {jr['job_role']}")
            if jr["probabilities"]:
                top3 = sorted(jr["probabilities"].items(), key=lambda x: -x[1])[:3]
                print("      Top 3    : " + "  |  ".join(
                    f"{r} {v:.0%}" for r, v in top3))
            print(f"    Salary     : ₹{sal['salary_lpa']:.2f} LPA")
        else:
            print("    (Salary & Job Role predictions skipped — not placed)")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
