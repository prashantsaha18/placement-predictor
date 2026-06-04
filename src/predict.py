"""
predict.py — Reusable prediction functions for all three models
Software Career Predictor

Usage (standalone):
    python predict.py

Or import into Streamlit / any downstream app:
    from predict import predict_placement, predict_job_role, predict_salary, predict_all
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from utils import (
    PLACEMENT_MODEL_PATH, JOB_ROLE_MODEL_PATH,
    SALARY_MODEL_PATH, ENCODER_PATH,
    NUMERIC_COLS, CATEGORICAL_COLS,
    print_section,
)


# ─────────────────────────────────────────────────────────────────────────────
# Load artefacts (lazy, cached in module-level variables)
# ─────────────────────────────────────────────────────────────────────────────
_placement_model = None
_job_role_model  = None
_salary_model    = None
_preprocessor    = None


def _load_models():
    global _placement_model, _job_role_model, _salary_model, _preprocessor
    if _placement_model is None:
        _placement_model = joblib.load(PLACEMENT_MODEL_PATH)
    if _job_role_model is None:
        _job_role_model  = joblib.load(JOB_ROLE_MODEL_PATH)
    if _salary_model is None:
        _salary_model    = joblib.load(SALARY_MODEL_PATH)
    if _preprocessor is None:
        _preprocessor    = joblib.load(ENCODER_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a single-row feature DataFrame from raw inputs
# ─────────────────────────────────────────────────────────────────────────────
def _build_input_df(
    cgpa: float,
    python_skill: int,
    dsa_skill: int,
    ml_skill: int,
    web_dev_skill: int,
    coding_score: float,
    communication_score: float,
    aptitude_score: float,
    internships: int,
    projects: int,
    backlogs: int,
    resume_score: float,
    skill_score: int,
    branch: str = "CSE",
    college_tier: int = 1,
) -> pd.DataFrame:
    """Return a single-row DataFrame with the same columns the preprocessor expects."""
    data = {
        # Numeric
        "cgpa":                [float(cgpa)],
        "python_skill":        [int(python_skill)],
        "dsa_skill":           [int(dsa_skill)],
        "ml_skill":            [int(ml_skill)],
        "web_dev_skill":       [int(web_dev_skill)],
        "coding_score":        [float(coding_score)],
        "communication_score": [float(communication_score)],
        "aptitude_score":      [float(aptitude_score)],
        "internships":         [int(internships)],
        "projects":            [int(projects)],
        "backlogs":            [int(backlogs)],
        "resume_score":        [float(resume_score)],
        "skill_score":         [int(skill_score)],
        # Categorical
        "branch":              [str(branch)],
        "college_tier":        [str(college_tier)],
    }
    return pd.DataFrame(data)[NUMERIC_COLS + CATEGORICAL_COLS]


def _transform_input(df: pd.DataFrame) -> np.ndarray:
    _load_models()
    return _preprocessor.transform(df)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def predict_placement(
    cgpa, python_skill, dsa_skill, ml_skill, web_dev_skill,
    coding_score, communication_score, aptitude_score,
    internships, projects, backlogs, resume_score, skill_score,
    branch="CSE", college_tier=1,
) -> dict:
    """
    Returns
    -------
    {
        "placed":       bool,
        "probability":  float  (0–1)
    }
    """
    _load_models()
    df  = _build_input_df(cgpa, python_skill, dsa_skill, ml_skill, web_dev_skill,
                           coding_score, communication_score, aptitude_score,
                           internships, projects, backlogs, resume_score, skill_score,
                           branch, college_tier)
    X   = _transform_input(df)
    prob = _placement_model.predict_proba(X)[0, 1]
    return {"placed": bool(prob >= 0.5), "probability": round(float(prob), 4)}


def predict_job_role(
    cgpa, python_skill, dsa_skill, ml_skill, web_dev_skill,
    coding_score, communication_score, aptitude_score,
    internships, projects, backlogs, resume_score, skill_score,
    branch="CSE", college_tier=1,
) -> dict:
    """
    Returns
    -------
    {
        "job_role":     str,
        "probabilities": dict[role -> float]
    }
    """
    _load_models()
    df  = _build_input_df(cgpa, python_skill, dsa_skill, ml_skill, web_dev_skill,
                           coding_score, communication_score, aptitude_score,
                           internships, projects, backlogs, resume_score, skill_score,
                           branch, college_tier)
    X   = _transform_input(df)
    idx = _job_role_model.predict(X)[0]

    # Recover class labels from the model
    classes = _job_role_model.classes_  # encoded integers
    # Map back to string labels stored in ENCODER_PATH's label encoder
    # (the le_role is not persisted separately; we use class ordering)
    from utils import JOB_ROLES
    role_labels = sorted(JOB_ROLES)           # alphabetical, matches LabelEncoder

    role_name   = role_labels[idx]

    proba_dict: dict = {}
    if hasattr(_job_role_model, "predict_proba"):
        probas = _job_role_model.predict_proba(X)[0]
        proba_dict = {role_labels[i]: round(float(p), 4) for i, p in enumerate(probas)}

    return {"job_role": role_name, "probabilities": proba_dict}


def predict_salary(
    cgpa, python_skill, dsa_skill, ml_skill, web_dev_skill,
    coding_score, communication_score, aptitude_score,
    internships, projects, backlogs, resume_score, skill_score,
    branch="CSE", college_tier=1,
) -> dict:
    """
    Returns
    -------
    {
        "salary_lpa": float
    }
    """
    _load_models()
    df  = _build_input_df(cgpa, python_skill, dsa_skill, ml_skill, web_dev_skill,
                           coding_score, communication_score, aptitude_score,
                           internships, projects, backlogs, resume_score, skill_score,
                           branch, college_tier)
    X   = _transform_input(df)
    sal = _salary_model.predict(X)[0]
    return {"salary_lpa": round(max(0.0, float(sal)), 2)}


def predict_all(
    cgpa, python_skill, dsa_skill, ml_skill, web_dev_skill,
    coding_score, communication_score, aptitude_score,
    internships, projects, backlogs, resume_score, skill_score,
    branch="CSE", college_tier=1,
) -> dict:
    """
    Run all three models and return a consolidated result dict.

    Returns
    -------
    {
        "placement":  { placed, probability },
        "job_role":   { job_role, probabilities },
        "salary":     { salary_lpa }
    }
    """
    common = dict(
        cgpa=cgpa, python_skill=python_skill, dsa_skill=dsa_skill,
        ml_skill=ml_skill, web_dev_skill=web_dev_skill,
        coding_score=coding_score, communication_score=communication_score,
        aptitude_score=aptitude_score, internships=internships,
        projects=projects, backlogs=backlogs, resume_score=resume_score,
        skill_score=skill_score, branch=branch, college_tier=college_tier,
    )
    return {
        "placement": predict_placement(**common),
        "job_role":  predict_job_role(**common),
        "salary":    predict_salary(**common),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Demo when run directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_section("PREDICTION DEMO — 3 sample students")

    samples = [
        dict(cgpa=8.9, python_skill=9, dsa_skill=8, ml_skill=8, web_dev_skill=7,
             coding_score=88, communication_score=85, aptitude_score=90,
             internships=2, projects=4, backlogs=0, resume_score=88, skill_score=85,
             branch="CSE", college_tier=1),
        dict(cgpa=6.2, python_skill=5, dsa_skill=4, ml_skill=3, web_dev_skill=6,
             coding_score=60, communication_score=65, aptitude_score=58,
             internships=1, projects=2, backlogs=2, resume_score=60, skill_score=55,
             branch="Mechanical", college_tier=3),
        dict(cgpa=7.5, python_skill=6, dsa_skill=7, ml_skill=5, web_dev_skill=8,
             coding_score=72, communication_score=70, aptitude_score=75,
             internships=1, projects=3, backlogs=0, resume_score=74, skill_score=68,
             branch="ECE", college_tier=2),
    ]

    labels = ["High-performer (CSE, Tier-1)",
              "Moderate (Mechanical, Tier-3)",
              "Good (ECE, Tier-2)"]

    for label, s in zip(labels, samples):
        result = predict_all(**s)
        p  = result["placement"]
        jr = result["job_role"]
        sal= result["salary"]
        print(f"\n  Student: {label}")
        print(f"    Placement  : {'✓ PLACED' if p['placed'] else '✗ NOT PLACED'}  "
              f"(probability={p['probability']:.2%})")
        if p["placed"]:
            print(f"    Job Role   : {jr['job_role']}")
            if jr["probabilities"]:
                top = sorted(jr["probabilities"].items(), key=lambda x: -x[1])[:3]
                print(f"      Top 3  : " + " | ".join(f"{r} {v:.1%}" for r, v in top))
            print(f"    Salary     : ₹{sal['salary_lpa']:.2f} LPA")
