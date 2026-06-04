"""
preprocess.py — Data loading, cleaning, encoding, and train/test splitting
Software Career Predictor
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from utils import (
    DATA_PATH, SCALER_PATH, ENCODER_PATH,
    NUMERIC_COLS, CATEGORICAL_COLS, ALL_FEATURE_COLS,
    TARGET_PLACEMENT, TARGET_JOB_ROLE, TARGET_SALARY,
    print_section, print_subsection,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load raw data
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Load the CSV and return a raw DataFrame."""
    df = pd.read_csv(DATA_PATH)
    print(f"[load_data] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Explore & summarise
# ─────────────────────────────────────────────────────────────────────────────
def explore_data(df: pd.DataFrame) -> None:
    """Print a concise EDA summary to stdout."""
    print_section("DATA EXPLORATION")

    print_subsection("Shape")
    print(f"  Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")

    print_subsection("Column info")
    print(df.dtypes.to_string())

    print_subsection("Missing values")
    mv = df.isnull().sum()
    mv = mv[mv > 0]
    if mv.empty:
        print("  None")
    else:
        print(mv.to_string())

    print_subsection("Duplicates")
    dups = df.duplicated().sum()
    print(f"  {dups} duplicate rows")

    print_subsection("Statistical summary (numeric)")
    print(df.describe(include=[np.number]).to_string())

    print_subsection("Target distributions")
    print("placed:")
    print(df[TARGET_PLACEMENT].value_counts().to_string())
    print("\njob_role:")
    print(df[TARGET_JOB_ROLE].value_counts(dropna=False).to_string())
    print("\nsalary_lpa (placed students):")
    sal = df.loc[df[TARGET_PLACEMENT] == 1, TARGET_SALARY]
    print(sal.describe().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 3. Clean
# ─────────────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values, duplicates, and outliers.

    Rules
    -----
    * Unplaced students have NaN for job_role / company_type — expected.
    * salary_lpa = 0 for unplaced students; keep as-is for regression
      (model will use placement subset).
    * Drop true duplicate rows (identical student_id).
    * Clip numeric feature outliers at [1st, 99th] percentile.
    """
    print_section("DATA CLEANING")
    orig_len = len(df)

    # Drop exact duplicates
    df = df.drop_duplicates()
    print(f"  Dropped {orig_len - len(df)} duplicate rows → {len(df):,} remain")

    # Clip numeric outliers (skill scores are already bounded 0-10/100 by design,
    # but cgpa / scores may have anomalies)
    for col in NUMERIC_COLS:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    print(f"  Clipped numeric outliers at [1st, 99th] percentile")

    # college_tier: cast to string for consistent encoding
    df["college_tier"] = df["college_tier"].astype(str)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Build preprocessor (scaler + encoder) and transform features
# ─────────────────────────────────────────────────────────────────────────────
def build_preprocessor(df: pd.DataFrame):
    """
    Fit a ColumnTransformer on df and return (preprocessor, X_transformed).

    The transformer:
      - StandardScaler  on NUMERIC_COLS
      - OneHotEncoder   on CATEGORICAL_COLS (branch, college_tier)

    The fitted object is saved to ENCODER_PATH so predict.py can reload it.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ]
    )

    X = df[ALL_FEATURE_COLS].copy()
    X_transformed = preprocessor.fit_transform(X)

    # Persist
    joblib.dump(preprocessor, ENCODER_PATH)
    print(f"  Saved preprocessor → {ENCODER_PATH}")
    print(f"  Feature matrix shape after encoding: {X_transformed.shape}")

    return preprocessor, X_transformed


def get_feature_names(preprocessor) -> list:
    """Return ordered feature names after ColumnTransformer."""
    num_names = NUMERIC_COLS
    cat_names = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLS)
    )
    return num_names + cat_names


# ─────────────────────────────────────────────────────────────────────────────
# 5. Encode job_role target
# ─────────────────────────────────────────────────────────────────────────────
def encode_job_role(series: pd.Series):
    """LabelEncode job_role. Returns (encoded_array, le_instance)."""
    le = LabelEncoder()
    encoded = le.fit_transform(series)
    return encoded, le


# ─────────────────────────────────────────────────────────────────────────────
# 6. Train / test splits
# ─────────────────────────────────────────────────────────────────────────────
def make_splits(df: pd.DataFrame, X_transformed: np.ndarray):
    """
    Return three (X_train, X_test, y_train, y_test) tuples for:
      placement  — all 9000 rows, binary target
      job_role   — only placed students (7702), multi-class target
      salary     — only placed students (7702), regression target

    Uses stratification where applicable.
    """
    print_section("TRAIN / TEST SPLITS  (80 / 20)")

    # ── Placement ────────────────────────────────────────────────────────────
    y_place = df[TARGET_PLACEMENT].values
    X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(
        X_transformed, y_place,
        test_size=0.20, random_state=42, stratify=y_place
    )
    print(f"  Placement  — train: {len(X_tr_p):,}  test: {len(X_te_p):,}")

    # ── Job Role & Salary — placed students only ──────────────────────────────
    placed_mask = df[TARGET_PLACEMENT].values == 1
    X_placed    = X_transformed[placed_mask]
    df_placed   = df[placed_mask].reset_index(drop=True)

    y_role_raw, le_role = encode_job_role(df_placed[TARGET_JOB_ROLE])
    y_salary            = df_placed[TARGET_SALARY].values

    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_placed, y_role_raw,
        test_size=0.20, random_state=42, stratify=y_role_raw
    )
    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
        X_placed, y_salary,
        test_size=0.20, random_state=42
    )
    print(f"  Job Role   — train: {len(X_tr_r):,}  test: {len(X_te_r):,}")
    print(f"  Salary     — train: {len(X_tr_s):,}  test: {len(X_te_s):,}")

    splits = {
        "placement": (X_tr_p, X_te_p, y_tr_p, y_te_p),
        "job_role":  (X_tr_r, X_te_r, y_tr_r, y_te_r),
        "salary":    (X_tr_s, X_te_s, y_tr_s, y_te_s),
    }
    return splits, le_role


# ─────────────────────────────────────────────────────────────────────────────
# 7. Master pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_preprocessing():
    """Full preprocessing pipeline. Returns (splits, preprocessor, le_role, feature_names, df)."""
    df = load_data()
    explore_data(df)
    df = clean_data(df)
    preprocessor, X_transformed = build_preprocessor(df)
    feature_names = get_feature_names(preprocessor)
    splits, le_role = make_splits(df, X_transformed)
    return splits, preprocessor, le_role, feature_names, df


if __name__ == "__main__":
    run_preprocessing()
    print("\n[preprocess.py] Done.")
