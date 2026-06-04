"""
utils.py — Shared constants, paths, and helper utilities
Software Career Predictor
"""

import os
import pathlib

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT_DIR / "data" / "student_placement_salary_elite_v2.csv"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Model persistence paths ───────────────────────────────────────────────────
PLACEMENT_MODEL_PATH = MODELS_DIR / "placement_model.pkl"
JOB_ROLE_MODEL_PATH  = MODELS_DIR / "job_role_model.pkl"
SALARY_MODEL_PATH    = MODELS_DIR / "salary_model.pkl"
SCALER_PATH          = MODELS_DIR / "scaler.pkl"
ENCODER_PATH         = MODELS_DIR / "encoder.pkl"

# ── Feature definitions ───────────────────────────────────────────────────────
# Categorical columns that get one-hot encoded
CATEGORICAL_COLS = ["branch", "college_tier"]

# Numeric features (used by scaler)
NUMERIC_COLS = [
    "cgpa", "python_skill", "dsa_skill", "ml_skill", "web_dev_skill",
    "coding_score", "communication_score", "aptitude_score",
    "internships", "projects", "backlogs", "resume_score", "skill_score",
]

# All features used by the models (order must match encoder output)
ALL_FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# Target columns
TARGET_PLACEMENT = "placed"
TARGET_JOB_ROLE  = "job_role"
TARGET_SALARY    = "salary_lpa"

# Job roles (for label decoding in predictions)
JOB_ROLES = ["Analyst", "Data Scientist", "Software Engineer", "Web Developer"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def print_section(title: str) -> None:
    """Print a formatted section header."""
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def print_subsection(title: str) -> None:
    print(f"\n--- {title} ---")
