# 🎯 Software Career Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://placement-predictor.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete, production-ready ML pipeline that predicts **placement probability**, **job role**, and **expected salary** for software engineering students — with a rich interactive Streamlit dashboard.

## ✨ New Features (v2)

| Feature | Description |
|---------|-------------|
| 🧑‍🏫 **Smart Career Advisor** | 5 personalized, actionable tips based on your weak spots |
| 🔄 **What-If Simulator** | See how improving CGPA, adding internships or skills changes your probability |
| 🏆 **Percentile Badge** | Bronze / Silver / Gold / Platinum tier with dataset percentile rank |
| 📊 **Dataset Explorer** | Full interactive dataset analysis across 9,000 students |
| 📥 **Download Report** | Export your full career report as a `.txt` file |
| ✨ **Smooth Animations** | Spinner + animated card reveals on every prediction |

---

## Project Structure

```
SoftwareCareerPredictor/
├── data/
│   └── student_placement_salary_elite_v2.csv   # 9,000-row dataset
├── models/                                      # Saved artefacts (auto-generated)
│   ├── placement_model.pkl
│   ├── job_role_model.pkl
│   ├── salary_model.pkl
│   └── encoder.pkl
├── plots/                                       # All visualisations (auto-generated)
├── src/
│   ├── main.py              # Master pipeline runner  ← START HERE
│   ├── preprocess.py        # Loading, cleaning, encoding, splitting
│   ├── eda_plots.py         # 7 exploratory visualisation charts
│   ├── train_placement.py   # Model 1: binary classification
│   ├── train_job_role.py    # Model 2: multi-class classification
│   ├── train_salary.py      # Model 3: regression
│   ├── predict.py           # Reusable inference functions
│   └── utils.py             # Shared constants & helpers
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
cd SoftwareCareerPredictor
python src/main.py
```

This single command:
- Generates 7 EDA plots
- Preprocesses and splits the data (80/20)
- Trains 4 placement models, selects best by F1
- Trains 3 job role models, selects best by Accuracy
- Trains 4 salary models, selects best by R²
- Saves all 4 model/encoder `.pkl` files
- Saves 17 visualisation `.png` files
- Runs a live demo on 3 sample students

### 3. Use predictions in your code

```python
import sys
sys.path.insert(0, "src")
from predict import predict_all, predict_placement, predict_job_role, predict_salary

# Single student profile
result = predict_all(
    cgpa=8.5, python_skill=1, dsa_skill=1, ml_skill=1, web_dev_skill=0,
    coding_score=82, communication_score=78, aptitude_score=85,
    internships=2, projects=3, backlogs=0, resume_score=80, skill_score=75,
    branch="CSE", college_tier=1,
)

print(result["placement"])  # {'placed': True, 'probability': 0.97}
print(result["job_role"])   # {'job_role': 'Data Scientist', 'probabilities': {...}}
print(result["salary"])     # {'salary_lpa': 82.4}
```

---

## Dataset

| Column | Type | Description |
|--------|------|-------------|
| `cgpa` | float | Cumulative GPA (5.0–10.0) |
| `branch` | str | Engineering branch (CSE, ECE, IT, EEE, Civil, Mechanical) |
| `college_tier` | int | College ranking tier (1, 2, 3) |
| `python_skill` | int | Python proficiency flag (0/1) |
| `dsa_skill` | int | DSA proficiency flag (0/1) |
| `ml_skill` | int | ML proficiency flag (0/1) |
| `web_dev_skill` | int | Web dev proficiency flag (0/1) |
| `coding_score` | float | Competitive coding score (0–100) |
| `communication_score` | float | Communication assessment (4–10) |
| `aptitude_score` | float | Aptitude test score (40–100) |
| `internships` | int | Number of internships (0–3) |
| `projects` | int | Number of projects (1–6) |
| `backlogs` | int | Academic backlogs (0–3) |
| `resume_score` | float | Resume evaluation score |
| `skill_score` | int | Composite skill score (0–4) |
| `placed` | int | **Target 1** — Placed (1) / Not placed (0) |
| `job_role` | str | **Target 2** — Analyst / Data Scientist / Software Engineer / Web Developer |
| `salary_lpa` | float | **Target 3** — Annual salary in LPA |

**Dataset facts:** 9,000 students · 85.6% placement rate · 4 balanced job roles · Salary range ₹32–129 LPA

---

## Model Results

### Model 1 — Placement Prediction (Binary Classification)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| **Logistic Regression** ✓ | 0.9256 | 0.9433 | 0.9714 | **0.9571** | **0.9716** |
| Gradient Boosting | 0.9117 | 0.9324 | 0.9669 | 0.9493 | 0.9563 |
| Random Forest | 0.9078 | 0.9139 | 0.9851 | 0.9481 | 0.9574 |
| Decision Tree | 0.8883 | 0.9272 | 0.9435 | 0.9353 | 0.8444 |

Best: **Logistic Regression** · F1 = 0.9571 · AUC = 0.9716

### Model 2 — Job Role Prediction (Multi-Class Classification)

> **Note:** In this synthetic dataset, the four job roles (Analyst, Data Scientist, Software Engineer, Web Developer) are **uniformly distributed** with near-zero correlation to any feature. The mean skill scores and salary distributions are statistically identical across all roles, confirming that job assignment is random in the data generation. All models converge to ~25% accuracy (equal to random chance on 4 balanced classes). This is a data limitation, not a model limitation — real-world datasets with genuine role-to-skill mappings yield 70–85% accuracy on this task.

| Model | Accuracy |
|-------|----------|
| Decision Tree | 0.2518 |
| Gradient Boosting | 0.2505 |
| Random Forest | 0.2408 |

### Model 3 — Salary Prediction (Regression)

| Model | R² | MAE (LPA) | RMSE (LPA) |
|-------|----|-----------|------------|
| **Random Forest** ✓ | **0.6218** | 7.95 | 10.24 |
| Gradient Boosting | 0.6081 | 8.13 | 10.43 |
| Linear Regression | 0.4391 | 10.07 | 12.48 |
| Ridge Regression | 0.4391 | 10.07 | 12.48 |

Best: **Random Forest** · R² = 0.6218 · MAE = ₹7.95 LPA

---

## Saved Artefacts

| File | Description |
|------|-------------|
| `models/placement_model.pkl` | Logistic Regression (binary classifier) |
| `models/job_role_model.pkl` | Decision Tree (multi-class classifier) |
| `models/salary_model.pkl` | Random Forest Regressor |
| `models/encoder.pkl` | ColumnTransformer (StandardScaler + OneHotEncoder) |

All saved with `joblib` for efficient loading. The encoder must be loaded alongside any model for consistent feature transformation.

---

## Visualisations Generated

**EDA (7 charts)**
- `eda_placement_distribution.png` — Pie chart + placement by college tier
- `eda_job_role_distribution.png` — Job role counts + breakdown by branch
- `eda_salary_distribution.png` — Histogram + box by role + violin by tier
- `eda_correlation_heatmap.png` — Lower-triangle correlation matrix (all numeric features)
- `eda_feature_distributions.png` — Placed vs Not Placed overlaid histograms for all 13 features
- `eda_cgpa_vs_salary.png` — CGPA vs salary scatter (coloured by coding score) + salary by internships
- `eda_skill_radar.png` — Polar chart of average skill profile per job role

**Placement (3 charts)**
- `placement_confusion_matrix.png`
- `placement_roc_curves.png` — All 4 models on one axes
- `placement_metrics_comparison.png` — Grouped bar chart

**Job Role (3 charts)**
- `job_role_confusion_matrix.png` — Counts + % side-by-side
- `job_role_feature_importance.png`
- `job_role_accuracy_comparison.png`

**Salary (4 charts)**
- `salary_actual_vs_predicted.png`
- `salary_residuals.png` — Residuals vs predicted + distribution
- `salary_feature_importance.png`
- `salary_metrics_comparison.png`

---

## Streamlit Integration

The `predict.py` API is Streamlit-ready. A minimal app:

```python
# app.py
import sys
sys.path.insert(0, "src")
import streamlit as st
from predict import predict_all

st.title("Software Career Predictor")

with st.sidebar:
    cgpa   = st.slider("CGPA", 5.0, 10.0, 7.5, 0.1)
    branch = st.selectbox("Branch", ["CSE", "ECE", "IT", "EEE", "Civil", "Mechanical"])
    tier   = st.selectbox("College Tier", [1, 2, 3])
    python = st.checkbox("Python skill")
    dsa    = st.checkbox("DSA skill")
    ml     = st.checkbox("ML skill")
    web    = st.checkbox("Web Dev skill")
    coding = st.slider("Coding Score", 0, 100, 65)
    comm   = st.slider("Communication Score", 4.0, 10.0, 7.0, 0.5)
    apt    = st.slider("Aptitude Score", 40, 100, 70)
    intern = st.number_input("Internships", 0, 3, 1)
    proj   = st.number_input("Projects", 1, 6, 3)
    back   = st.number_input("Backlogs", 0, 3, 0)
    resume = st.slider("Resume Score", 0, 100, 70)
    skill  = st.slider("Skill Score", 0, 4, 2)

if st.button("Predict"):
    r = predict_all(cgpa, int(python), int(dsa), int(ml), int(web),
                    coding, comm, apt, intern, proj, back, resume, skill,
                    branch, tier)
    p = r["placement"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Placement", "✅ Placed" if p["placed"] else "❌ Not Placed",
                f"{p['probability']:.1%}")
    col2.metric("Job Role", r["job_role"]["job_role"])
    col3.metric("Expected Salary", f"₹{r['salary']['salary_lpa']} LPA")
```

Run with: `streamlit run app.py`

---

## Extending the Pipeline

**Add XGBoost** (when available):
```python
# In train_placement.py / train_job_role.py / train_salary.py
from xgboost import XGBClassifier, XGBRegressor
MODELS["XGBoost"] = XGBClassifier(n_estimators=200, learning_rate=0.1,
                                   max_depth=5, use_label_encoder=False,
                                   eval_metric="logloss", random_state=42)
```

**Hyperparameter tuning:**
```python
from sklearn.model_selection import RandomizedSearchCV
param_grid = {"n_estimators": [100, 200, 300], "max_depth": [5, 8, 12]}
search = RandomizedSearchCV(RandomForestClassifier(), param_grid, n_iter=10, cv=5)
search.fit(X_train, y_train)
best = search.best_estimator_
```
