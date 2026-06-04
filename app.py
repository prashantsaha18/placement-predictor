"""
app.py — Streamlit frontend for Software Career Predictor
Run: streamlit run app.py
"""

import sys
import pathlib
import warnings
import io
import datetime
warnings.filterwarnings("ignore")

# ── Make src/ importable ──────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Software Career Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #13161e;
    border-right: 1px solid #1f2330;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #7c83f5;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    letter-spacing: 0.03em;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a1d2e 0%, #0d1526 50%, #0a1a1a 100%);
    border: 1px solid #1f2a3a;
    border-radius: 16px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(124,131,245,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.hero-subtitle {
    color: #7c83f5;
    font-size: 1rem;
    font-weight: 500;
    margin: 0;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Result cards ── */
.result-card {
    background: #13161e;
    border: 1px solid #1f2330;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.result-card.placed  { border-left: 4px solid #22c55e; }
.result-card.not-placed { border-left: 4px solid #ef4444; }
.result-card.role    { border-left: 4px solid #7c83f5; }
.result-card.salary  { border-left: 4px solid #f59e0b; }

.result-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.4rem;
}
.result-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
}
.result-sub {
    font-size: 0.85rem;
    color: #9ca3af;
    margin-top: 0.3rem;
}

/* ── Confidence bar ── */
.conf-bar-wrap { margin-top: 0.8rem; }
.conf-bar-bg {
    background: #1f2330;
    border-radius: 99px;
    height: 6px;
    width: 100%;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}

/* ── Section headers ── */
.section-head {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #c7caff;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border-bottom: 1px solid #1f2330;
    padding-bottom: 0.5rem;
    margin: 1.6rem 0 1rem 0;
}

/* ── Insight chips ── */
.chip {
    display: inline-block;
    background: #1a1d2e;
    border: 1px solid #2a2f45;
    border-radius: 99px;
    padding: 0.2rem 0.75rem;
    font-size: 0.8rem;
    color: #a5b4fc;
    margin: 0.2rem;
}

/* ── Metric row ── */
.metric-pill {
    background: #13161e;
    border: 1px solid #1f2330;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    text-align: center;
}
.metric-pill .num {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #7c83f5;
}
.metric-pill .lbl {
    font-size: 0.72rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Advisor cards ── */
.advisor-card {
    background: linear-gradient(135deg, #13161e, #1a1d2e);
    border: 1px solid #2a2f45;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    animation: fadeSlideIn 0.4s ease both;
}
.advisor-icon {
    font-size: 1.4rem;
    flex-shrink: 0;
    margin-top: 0.1rem;
}
.advisor-text { flex: 1; }
.advisor-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: #c7caff;
    margin-bottom: 0.2rem;
}
.advisor-body {
    font-size: 0.83rem;
    color: #9ca3af;
    line-height: 1.5;
}

/* ── Percentile Badge ── */
.badge-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1.5rem;
    background: #13161e;
    border-radius: 14px;
    border: 1px solid #1f2330;
    margin-bottom: 1rem;
}
.badge-circle {
    width: 110px; height: 110px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-direction: column;
    font-family: 'Syne', sans-serif;
    position: relative;
    margin-bottom: 0.8rem;
}
.badge-score {
    font-size: 1.8rem;
    font-weight: 800;
    color: white;
    line-height: 1;
}
.badge-label {
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.7);
}
.badge-tier {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-pct {
    font-size: 0.82rem;
    color: #6b7280;
    margin-top: 0.3rem;
}

/* ── What-If ── */
.whatif-card {
    background: #13161e;
    border: 1px solid #1f2330;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.6rem;
}

/* ── Animations ── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes glow-pulse {
    0%, 100% { box-shadow: 0 0 18px rgba(124,131,245,0.35); }
    50%       { box-shadow: 0 0 32px rgba(124,131,245,0.65); }
}
.badge-circle.platinum { animation: glow-pulse 2.4s ease-in-out infinite; }

/* ── Divider ── */
hr { border-color: #1f2330; margin: 1.2rem 0; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #7c83f5, #5b63e8);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.04em;
    padding: 0.7rem 1.4rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* ── Slider / input labels ── */
label { color: #9ca3af !important; font-size: 0.85rem !important; }
.stSlider > div > div > div > div { background: #7c83f5 !important; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: #13161e;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #6b7280;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #1f2330 !important;
    color: #c7caff !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1a2a1a, #1a3a1a);
    border: 1px solid #22c55e;
    color: #22c55e !important;
}
.stDownloadButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = ROOT / "models"
    return {
        "placement": joblib.load(base / "placement_model.pkl"),
        "job_role":  joblib.load(base / "job_role_model.pkl"),
        "salary":    joblib.load(base / "salary_model.pkl"),
        "encoder":   joblib.load(base / "encoder.pkl"),
    }

try:
    models = load_models()
    MODELS_READY = True
except Exception as e:
    MODELS_READY = False
    MODEL_ERROR = str(e)


# ── Load dataset (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    csv_path = ROOT / "student_placement_salary_elite_v2.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

DATASET = load_dataset()


# ── Inference helpers ─────────────────────────────────────────────────────────
NUMERIC_COLS = [
    "cgpa","python_skill","dsa_skill","ml_skill","web_dev_skill",
    "coding_score","communication_score","aptitude_score",
    "internships","projects","backlogs","resume_score","skill_score",
]
CATEGORICAL_COLS = ["branch", "college_tier"]
JOB_ROLES = sorted(["Analyst","Data Scientist","Software Engineer","Web Developer"])
ROLE_ICONS = {"Analyst":"📊","Data Scientist":"🧠","Software Engineer":"💻","Web Developer":"🌐"}


def predict(inputs: dict) -> dict:
    row = {c: [inputs[c]] for c in NUMERIC_COLS + CATEGORICAL_COLS}
    df  = pd.DataFrame(row)
    X   = models["encoder"].transform(df)

    place_prob  = float(models["placement"].predict_proba(X)[0, 1])
    placed      = place_prob >= 0.5

    role_idx    = int(models["job_role"].predict(X)[0])
    role_name   = JOB_ROLES[role_idx]
    role_probas = {}
    if hasattr(models["job_role"], "predict_proba"):
        rp = models["job_role"].predict_proba(X)[0]
        role_probas = {JOB_ROLES[i]: float(p) for i, p in enumerate(rp)}

    salary = float(models["salary"].predict(X)[0])

    return {
        "placed":       placed,
        "prob":         place_prob,
        "role":         role_name,
        "role_probas":  role_probas,
        "salary":       max(0.0, salary),
    }


# ── Percentile & Readiness Score ──────────────────────────────────────────────
def compute_readiness(inputs: dict, prob: float) -> dict:
    """Composite 0-100 readiness score with tier + percentile estimate."""
    cgpa_norm    = (inputs["cgpa"] - 5.0) / 5.0 * 25          # max 25
    prob_norm    = prob * 30                                    # max 30
    skill_norm   = inputs["skill_score"] / 4 * 15              # max 15
    exp_norm     = min(inputs["internships"] / 3, 1) * 15      # max 15
    score_norm   = (inputs["coding_score"] / 100 * 5 +
                    inputs["resume_score"] / 100 * 5 +
                    (inputs["aptitude_score"] - 40) / 60 * 5)  # max 15 (but broken into 5+5+5)
    total = cgpa_norm + prob_norm + skill_norm + exp_norm + score_norm
    total = min(100.0, max(0.0, total))

    # Estimate percentile from a simplified CDF (dataset mean ~62, std ~14)
    from scipy import stats as scipy_stats
    pct = float(scipy_stats.norm.cdf(total, loc=62, scale=14)) * 100

    if total >= 85:
        tier, color, glow = "Platinum", "#e2e8f0", "platinum"
    elif total >= 70:
        tier, color, glow = "Gold", "#f59e0b", "gold"
    elif total >= 55:
        tier, color, glow = "Silver", "#94a3b8", "silver"
    else:
        tier, color, glow = "Bronze", "#b45309", "bronze"

    return {"score": total, "pct": pct, "tier": tier, "color": color, "glow": glow}


# ── Smart Career Advisor ──────────────────────────────────────────────────────
def get_advisor_tips(inputs: dict, res: dict) -> list:
    tips = []

    if inputs["cgpa"] < 7.0:
        tips.append(("📚", "Improve Your CGPA",
                     f"Your CGPA of {inputs['cgpa']:.1f} is below 7.0. Many top companies have a 7.0 cutoff. "
                     "Focus on core subjects and aim for consistent performance this semester."))

    if inputs["internships"] == 0:
        tips.append(("🏢", "Get an Internship",
                     "You have no internship experience yet — this is one of the strongest placement signals. "
                     "Apply to Summer internship programs (FAANG, startups, research labs) immediately."))
    elif inputs["internships"] == 1:
        tips.append(("🏢", "Add a Second Internship",
                     "One internship is good, but students with 2+ internships have 18% higher placement rates. "
                     "Consider a part-time or remote internship alongside your studies."))

    if inputs["skill_score"] < 2:
        tips.append(("🛠️", "Build More Technical Skills",
                     "You currently have fewer than 2 skills active (Python, DSA, ML, Web Dev). "
                     "Even 1 weekend learning Python + 2 weeks on DSA can dramatically improve your profile."))

    if inputs["dsa_skill"] == 0:
        tips.append(("🧩", "Learn Data Structures & Algorithms",
                     "DSA is tested in virtually every tech interview. "
                     "Practice 2–3 LeetCode problems daily on Arrays, Strings, and Trees for 60 days."))

    if inputs["coding_score"] < 55:
        tips.append(("💻", "Improve Your Coding Score",
                     f"Your coding score ({inputs['coding_score']}) is below median (49.8 dataset avg, but top candidates score 75+). "
                     "Participate in Codeforces/HackerRank weekly contests to build speed."))

    if inputs["resume_score"] < 60:
        tips.append(("📄", "Strengthen Your Resume",
                     f"Your resume score of {inputs['resume_score']} has room to grow. "
                     "Use a clean single-page format, quantify achievements (e.g., '↑ system speed by 35%'), "
                     "and get it reviewed on r/cscareerquestions."))

    if inputs["projects"] < 3:
        tips.append(("🚀", "Build More Projects",
                     f"You have {inputs['projects']} project(s). Aim for 3–4 showcasing different skills. "
                     "A deployed web app, an ML pipeline on Kaggle, and a DSA-focused GitHub repo make a strong combo."))

    if inputs["backlogs"] > 0:
        tips.append(("⚠️", "Clear Your Backlogs",
                     f"You have {inputs['backlogs']} active backlog(s). These are flagged in 80% of company background checks. "
                     "Prioritize clearing them before campus placements open."))

    if res["prob"] >= 0.85:
        tips.insert(0, ("🌟", "You're in Great Shape!",
                        f"Your placement probability is {res['prob']*100:.1f}%. Focus on maintaining your profile, "
                        "applying to dream companies early, and negotiating your offer package."))

    return tips[:5]  # Return top 5 most relevant tips


# ── What-If Simulator ─────────────────────────────────────────────────────────
def compute_whatif(base_inputs: dict, base_prob: float) -> list:
    scenarios = []

    # Scenario 1: CGPA +0.5
    if base_inputs["cgpa"] < 10.0:
        inp2 = dict(base_inputs, cgpa=min(10.0, base_inputs["cgpa"] + 0.5))
        r2 = predict(inp2)
        delta = r2["prob"] - base_prob
        scenarios.append(("📈 CGPA +0.5", delta, r2["prob"]))

    # Scenario 2: +1 Internship
    if base_inputs["internships"] < 3:
        inp2 = dict(base_inputs, internships=base_inputs["internships"] + 1)
        r2 = predict(inp2)
        delta = r2["prob"] - base_prob
        scenarios.append(("🏢 +1 Internship", delta, r2["prob"]))

    # Scenario 3: +1 Project
    if base_inputs["projects"] < 6:
        inp2 = dict(base_inputs, projects=base_inputs["projects"] + 1)
        r2 = predict(inp2)
        delta = r2["prob"] - base_prob
        scenarios.append(("🚀 +1 Project", delta, r2["prob"]))

    # Scenario 4: Add DSA
    if base_inputs["dsa_skill"] == 0:
        inp2 = dict(base_inputs, dsa_skill=1,
                    skill_score=min(4, base_inputs["skill_score"] + 1))
        r2 = predict(inp2)
        delta = r2["prob"] - base_prob
        scenarios.append(("🧩 Learn DSA", delta, r2["prob"]))

    # Scenario 5: Add Python
    if base_inputs["python_skill"] == 0:
        inp2 = dict(base_inputs, python_skill=1,
                    skill_score=min(4, base_inputs["skill_score"] + 1))
        r2 = predict(inp2)
        delta = r2["prob"] - base_prob
        scenarios.append(("🐍 Learn Python", delta, r2["prob"]))

    # Scenario 6: Coding Score +15
    if base_inputs["coding_score"] <= 85:
        inp2 = dict(base_inputs, coding_score=min(100, base_inputs["coding_score"] + 15))
        r2 = predict(inp2)
        delta = r2["prob"] - base_prob
        scenarios.append(("💻 Coding +15pts", delta, r2["prob"]))

    # Scenario 7: Clear Backlogs
    if base_inputs["backlogs"] > 0:
        inp2 = dict(base_inputs, backlogs=0)
        r2 = predict(inp2)
        delta = r2["prob"] - base_prob
        scenarios.append(("✅ Clear Backlogs", delta, r2["prob"]))

    # Sort by most impactful (highest positive delta first)
    scenarios.sort(key=lambda x: -x[1])
    return scenarios


# ── Generate download report ──────────────────────────────────────────────────
def generate_report(inputs: dict, res: dict, readiness: dict) -> str:
    date_str = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")
    skills = []
    for skill, key in [("Python","python_skill"),("DSA","dsa_skill"),
                       ("Machine Learning","ml_skill"),("Web Development","web_dev_skill")]:
        if inputs[key]:
            skills.append(skill)

    lines = [
        "=" * 60,
        "  SOFTWARE CAREER PREDICTOR — CAREER REPORT",
        f"  Generated: {date_str}",
        "=" * 60,
        "",
        "STUDENT PROFILE",
        "-" * 40,
        f"  CGPA            : {inputs['cgpa']:.1f}",
        f"  Branch          : {inputs['branch']}",
        f"  College Tier    : Tier {inputs['college_tier']}",
        f"  Skills          : {', '.join(skills) if skills else 'None selected'}",
        f"  Internships     : {inputs['internships']}",
        f"  Projects        : {inputs['projects']}",
        f"  Backlogs        : {inputs['backlogs']}",
        f"  Coding Score    : {inputs['coding_score']}",
        f"  Communication   : {inputs['communication_score']}",
        f"  Aptitude Score  : {inputs['aptitude_score']}",
        f"  Resume Score    : {inputs['resume_score']}",
        "",
        "PREDICTION RESULTS",
        "-" * 40,
        f"  Placement       : {'✓ PLACED' if res['placed'] else '✗ NOT PLACED'}",
        f"  Probability     : {res['prob']*100:.1f}%",
        f"  Predicted Role  : {res['role']}",
        f"  Expected Salary : ₹{res['salary']:.1f} LPA",
        "",
        "READINESS ASSESSMENT",
        "-" * 40,
        f"  Readiness Score : {readiness['score']:.1f} / 100",
        f"  Tier            : {readiness['tier']}",
        f"  Percentile      : Top {100 - readiness['pct']:.0f}% of dataset",
        "",
        "JOB ROLE PROBABILITIES",
        "-" * 40,
    ]
    for role, prob in sorted(res["role_probas"].items(), key=lambda x: -x[1]):
        lines.append(f"  {role:<22}: {prob*100:.1f}%")

    lines += [
        "",
        "PERSONALIZED TIPS",
        "-" * 40,
    ]
    tips = get_advisor_tips(inputs, res)
    for i, (_, title, body) in enumerate(tips, 1):
        lines.append(f"  {i}. {title}")
        lines.append(f"     {body}")
        lines.append("")

    lines += [
        "=" * 60,
        "  Software Career Predictor · Trained on 9,000 students",
        "  3 ML Models · Built with Streamlit & scikit-learn",
        "=" * 60,
    ]
    return "\n".join(lines)


# ── Sidebar — Input form ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Student Profile")
    st.markdown("---")

    st.markdown("### 🏫 Academic")
    cgpa   = st.slider("CGPA", 5.0, 10.0, 7.8, 0.1)
    branch = st.selectbox("Branch", ["CSE","ECE","IT","EEE","Civil","Mechanical"])
    tier   = st.selectbox("College Tier", [1, 2, 3], format_func=lambda x: f"Tier {x}")
    backlogs = st.slider("Backlogs", 0, 3, 0)

    st.markdown("### 🛠️ Technical Skills")
    python  = st.toggle("Python",  value=True)
    dsa     = st.toggle("DSA",     value=True)
    ml      = st.toggle("Machine Learning", value=False)
    webdev  = st.toggle("Web Development",  value=False)

    st.markdown("### 📊 Assessment Scores")
    coding  = st.slider("Coding Score",        0,   100, 72)
    comm    = st.slider("Communication Score", 4.0, 10.0, 7.0, 0.5)
    apt     = st.slider("Aptitude Score",      40,  100, 75)
    resume  = st.slider("Resume Score",        0,   100, 70)

    st.markdown("### 💼 Experience")
    internships = st.selectbox("Internships", [0,1,2,3])
    projects    = st.selectbox("Projects",    [1,2,3,4,5,6])

    # Derived skill score
    skill_score = int(python) + int(dsa) + int(ml) + int(webdev)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Career Outcome", use_container_width=True)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <p class="hero-subtitle">ML-Powered · 3 Models · 9,000 Students</p>
  <h1 class="hero-title">Software Career<br>Predictor</h1>
</div>
""", unsafe_allow_html=True)

if not MODELS_READY:
    st.error(f"⚠️ Could not load models: {MODEL_ERROR}\n\nRun `python src/main.py` first to train and save the models.")
    st.stop()

# ── Build inputs dict ─────────────────────────────────────────────────────────
inputs = dict(
    cgpa=cgpa, python_skill=int(python), dsa_skill=int(dsa),
    ml_skill=int(ml), web_dev_skill=int(webdev),
    coding_score=float(coding), communication_score=float(comm),
    aptitude_score=float(apt), internships=internships,
    projects=projects, backlogs=backlogs,
    resume_score=float(resume), skill_score=skill_score,
    branch=branch, college_tier=str(tier),
)

# ── Run prediction ────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    with st.spinner("🔮 Running prediction models…"):
        st.session_state.result = predict(inputs)

if predict_btn:
    with st.spinner("🔮 Running prediction models…"):
        import time; time.sleep(0.6)   # brief dramatic pause
        st.session_state.result = predict(inputs)

res = st.session_state.result

# ────────────────────────────────────────────────────────────────────────────
# Results panel — 4 tabs
# ────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  🎯  Prediction Results  ",
    "  📈  Analysis & Charts  ",
    "  🔄  What-If Simulator  ",
    "  📊  Dataset Explorer  ",
])

# ============================================================================
# TAB 1 — Prediction Results
# ============================================================================
with tab1:
    col_a, col_b, col_c = st.columns(3)

    # ── Card 1: Placement ─────────────────────────────────────────────────────
    with col_a:
        placed_class = "placed" if res["placed"] else "not-placed"
        placed_icon  = "✅" if res["placed"] else "❌"
        placed_label = "PLACED" if res["placed"] else "NOT PLACED"
        bar_color    = "#22c55e" if res["placed"] else "#ef4444"
        conf_pct     = res["prob"] * 100
        conf_display = f"{conf_pct:.1f}%"

        st.markdown(f"""
        <div class="result-card {placed_class}">
          <div class="result-label">Placement Prediction</div>
          <div class="result-value">{placed_icon} {placed_label}</div>
          <div class="result-sub">Confidence: {conf_display}</div>
          <div class="conf-bar-wrap">
            <div class="conf-bar-bg">
              <div class="conf-bar-fill" style="width:{conf_pct:.1f}%;background:{bar_color};"></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Card 2: Job Role ──────────────────────────────────────────────────────
    with col_b:
        role_icon = ROLE_ICONS.get(res["role"], "💼")
        top_role_conf = res["role_probas"].get(res["role"], 0) * 100
        st.markdown(f"""
        <div class="result-card role">
          <div class="result-label">Predicted Job Role</div>
          <div class="result-value">{role_icon} {res['role']}</div>
          <div class="result-sub">Model confidence: {top_role_conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Card 3: Salary ────────────────────────────────────────────────────────
    with col_c:
        sal = res["salary"]
        sal_tier = "🔥 Top Earner" if sal > 90 else ("💰 Above Average" if sal > 65 else "📦 Entry Level")
        st.markdown(f"""
        <div class="result-card salary">
          <div class="result-label">Expected Salary</div>
          <div class="result-value">₹{sal:.1f} LPA</div>
          <div class="result-sub">{sal_tier}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Readiness Badge + Profile Summary ─────────────────────────────────────
    badge_col, profile_col = st.columns([1, 2])

    with badge_col:
        try:
            readiness = compute_readiness(inputs, res["prob"])
        except ImportError:
            # scipy not available — fallback
            score = min(100, (res["prob"] * 30 + inputs["cgpa"] / 10 * 25 +
                              inputs["skill_score"] / 4 * 15 +
                              inputs["internships"] / 3 * 15 + 15))
            pct = score
            tier_map = [("Platinum","#e2e8f0","platinum"),("Gold","#f59e0b","gold"),
                        ("Silver","#94a3b8","silver"),("Bronze","#b45309","bronze")]
            tier, color, glow = tier_map[0] if score>=85 else (tier_map[1] if score>=70 else
                                (tier_map[2] if score>=55 else tier_map[3]))
            readiness = {"score": score, "pct": pct, "tier": tier, "color": color, "glow": glow}

        tier_bg = {
            "Platinum": "linear-gradient(135deg, #334155, #64748b)",
            "Gold":     "linear-gradient(135deg, #78350f, #d97706)",
            "Silver":   "linear-gradient(135deg, #334155, #64748b)",
            "Bronze":   "linear-gradient(135deg, #451a03, #92400e)",
        }.get(readiness["tier"], "linear-gradient(135deg, #1e293b, #334155)")

        st.markdown(f"""
        <div class="badge-wrap">
          <div class="badge-circle {readiness['glow']}" style="background:{tier_bg};">
            <div class="badge-score">{readiness['score']:.0f}</div>
            <div class="badge-label">/ 100</div>
          </div>
          <div class="badge-tier" style="color:{readiness['color']};">
            {readiness['tier']} Tier
          </div>
          <div class="badge-pct">Top {100 - readiness['pct']:.0f}% of students</div>
        </div>
        """, unsafe_allow_html=True)

    with profile_col:
        st.markdown("<div class='section-head'>Profile Summary</div>", unsafe_allow_html=True)

        skills_on = []
        if python: skills_on.append("Python")
        if dsa:    skills_on.append("DSA")
        if ml:     skills_on.append("Machine Learning")
        if webdev: skills_on.append("Web Development")

        col1, col2, col3, col4 = st.columns(4)
        for col, num, label in zip(
            [col1, col2, col3, col4],
            [f"{cgpa:.1f}", str(internships), str(projects), str(backlogs)],
            ["CGPA", "Internships", "Projects", "Backlogs"]
        ):
            with col:
                st.markdown(f"""
                <div class="metric-pill">
                  <div class="num">{num}</div>
                  <div class="lbl">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='margin-top:1rem;'>
          <span class='chip'>🏫 {branch}</span>
          <span class='chip'>🏆 Tier {tier}</span>
          {''.join(f"<span class='chip'>✔ {s}</span>" for s in skills_on)}
          <span class='chip'>📝 Resume {resume}</span>
          <span class='chip'>💬 Comm {comm}</span>
          <span class='chip'>🧮 Aptitude {apt}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Job role probabilities ────────────────────────────────────────────────
    if res["role_probas"]:
        st.markdown("<div class='section-head'>Job Role Probabilities</div>", unsafe_allow_html=True)
        sorted_roles = sorted(res["role_probas"].items(), key=lambda x: -x[1])
        for role, prob in sorted_roles:
            icon  = ROLE_ICONS.get(role, "💼")
            pct   = prob * 100
            is_top = role == res["role"]
            bar_c = "#7c83f5" if is_top else "#2a2f45"
            st.markdown(f"""
            <div style='margin-bottom:0.5rem;'>
              <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
                <span style='font-size:0.9rem;color:{"#c7caff" if is_top else "#6b7280"};'>
                  {icon} {role}{"  ◀" if is_top else ""}
                </span>
                <span style='font-size:0.85rem;color:#9ca3af;'>{pct:.1f}%</span>
              </div>
              <div class='conf-bar-bg'>
                <div class='conf-bar-fill' style='width:{pct:.1f}%;background:{bar_c};'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Smart Career Advisor ──────────────────────────────────────────────────
    st.markdown("<div class='section-head'>🧑‍🏫 Smart Career Advisor</div>", unsafe_allow_html=True)
    tips = get_advisor_tips(inputs, res)
    for icon, title, body in tips:
        st.markdown(f"""
        <div class="advisor-card">
          <div class="advisor-icon">{icon}</div>
          <div class="advisor-text">
            <div class="advisor-title">{title}</div>
            <div class="advisor-body">{body}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Download Report ───────────────────────────────────────────────────────
    try:
        readiness
    except NameError:
        readiness = {"score": 50.0, "pct": 50.0, "tier": "Silver", "color": "#94a3b8", "glow": "silver"}

    report_text = generate_report(inputs, res, readiness)
    st.download_button(
        label="📥 Download Career Report (.txt)",
        data=report_text.encode("utf-8"),
        file_name=f"career_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ============================================================================
# TAB 2 — Analysis & Charts
# ============================================================================
with tab2:
    st.markdown("<div class='section-head'>Skill Radar vs Dataset Average</div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])

    with col_l:
        skill_labels = ["Python", "DSA", "ML", "Web Dev"]
        student_vals = [int(python), int(dsa), int(ml), int(webdev)]
        avg_vals     = [0.65, 0.56, 0.30, 0.39]

        N = len(skill_labels)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        sv = student_vals + student_vals[:1]
        av = avg_vals + avg_vals[:1]

        fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#13161e")
        ax.set_facecolor("#0d0f14")

        ax.plot(angles, av, color="#374151", linewidth=1.5, linestyle="--", label="Dataset Avg")
        ax.fill(angles, av, color="#374151", alpha=0.15)
        ax.plot(angles, sv, color="#7c83f5", linewidth=2.5, label="You")
        ax.fill(angles, sv, color="#7c83f5", alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(skill_labels, color="#9ca3af", fontsize=10)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "", "", ""], color="#374151")
        ax.grid(color="#1f2330", linewidth=0.8)
        ax.spines["polar"].set_color("#1f2330")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15),
                  fontsize=9, labelcolor="#9ca3af", facecolor="#13161e", edgecolor="#1f2330")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_r:
        score_labels = ["Coding", "Comm ×10", "Aptitude", "Resume"]
        score_vals   = [coding, comm*10, apt, resume]
        avg_scores   = [49.8, 69.9, 70.1, 67.8]

        fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
        fig2.patch.set_facecolor("#13161e")
        ax2.set_facecolor("#0d0f14")

        x = np.arange(len(score_labels))
        w = 0.35
        ax2.bar(x - w/2, avg_scores, w, color="#2a2f45", label="Dataset Avg", zorder=3)
        ax2.bar(x + w/2, score_vals,  w, color="#7c83f5", label="You",        zorder=3)

        ax2.set_xticks(x)
        ax2.set_xticklabels(score_labels, color="#9ca3af", fontsize=9)
        ax2.set_ylim(0, 115)
        ax2.set_ylabel("Score", color="#6b7280", fontsize=9)
        ax2.tick_params(colors="#6b7280")
        ax2.spines[:].set_color("#1f2330")
        ax2.yaxis.label.set_color("#6b7280")
        ax2.grid(axis="y", color="#1f2330", linewidth=0.6, zorder=0)
        ax2.legend(fontsize=9, labelcolor="#9ca3af", facecolor="#13161e", edgecolor="#1f2330")
        ax2.set_title("Score Comparison vs Dataset", color="#9ca3af", fontsize=10, pad=10)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # ── Placement probability gauge ───────────────────────────────────────────
    st.markdown("<div class='section-head'>Placement Probability Gauge</div>", unsafe_allow_html=True)

    fig3, ax3 = plt.subplots(figsize=(7, 3.5))
    fig3.patch.set_facecolor("#13161e")
    ax3.set_facecolor("#13161e")

    prob = res["prob"]
    theta = np.linspace(np.pi, 0, 300)
    ax3.plot(np.cos(theta), np.sin(theta), color="#1f2330", linewidth=18, solid_capstyle="round")
    fill_theta = np.linspace(np.pi, np.pi - prob*np.pi, 300)
    grad_color = "#22c55e" if prob > 0.7 else ("#f59e0b" if prob > 0.4 else "#ef4444")
    ax3.plot(np.cos(fill_theta), np.sin(fill_theta), color=grad_color, linewidth=18, solid_capstyle="round")
    angle = np.pi - prob * np.pi
    ax3.annotate("", xy=(0.65*np.cos(angle), 0.65*np.sin(angle)), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="white", lw=2, mutation_scale=18))
    ax3.text(0, -0.22, f"{prob*100:.1f}%", ha="center", va="center",
             fontsize=28, fontweight="bold", color="white", fontfamily="sans-serif")
    ax3.text(0, -0.48, "Placement Probability", ha="center", color="#6b7280", fontsize=10)
    ax3.text(-1.05, -0.08, "0%", ha="center", color="#6b7280", fontsize=9)
    ax3.text( 1.05, -0.08, "100%", ha="center", color="#6b7280", fontsize=9)
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-0.65, 1.05)
    ax3.axis("off")
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # ── Salary benchmark ──────────────────────────────────────────────────────
    st.markdown("<div class='section-head'>Salary Benchmark</div>", unsafe_allow_html=True)

    sal_benchmarks = {
        "Entry (P25)":  49.0,
        "Median (P50)": 65.9,
        "Your Estimate": res["salary"],
        "Top (P75)":    77.5,
        "Elite (P90)":  90.0,
    }

    fig4, ax4 = plt.subplots(figsize=(9, 2.8))
    fig4.patch.set_facecolor("#13161e")
    ax4.set_facecolor("#13161e")

    labels = list(sal_benchmarks.keys())
    vals   = list(sal_benchmarks.values())
    colors = ["#374151","#4b5563","#7c83f5","#6b7280","#4b5563"]

    bars = ax4.barh(labels, vals, color=colors, edgecolor="#0d0f14", height=0.55)
    for bar, val in zip(bars, vals):
        ax4.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f"₹{val:.1f}", va="center", color="white", fontsize=9, fontweight="bold")
    ax4.set_xlim(0, 105)
    ax4.set_xlabel("Salary (LPA)", color="#6b7280", fontsize=9)
    ax4.tick_params(colors="#9ca3af", labelsize=9)
    ax4.spines[:].set_color("#1f2330")
    ax4.grid(axis="x", color="#1f2330", linewidth=0.6)
    ax4.set_title("Your Salary vs Dataset Percentiles", color="#9ca3af", fontsize=10, pad=8)
    plt.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)


# ============================================================================
# TAB 3 — What-If Simulator
# ============================================================================
with tab3:
    st.markdown("""
    <div style='background:#13161e;border:1px solid #1f2330;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1.4rem;'>
      <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#c7caff;margin-bottom:0.4rem;'>
        🔄 What-If Career Simulator
      </div>
      <div style='font-size:0.85rem;color:#6b7280;line-height:1.6;'>
        See exactly how improving one aspect of your profile would change your placement probability.
        Each bar shows the <b style='color:#9ca3af'>delta (change)</b> from your current probability of
        <b style='color:#7c83f5'>{:.1f}%</b>.
      </div>
    </div>
    """.format(res["prob"] * 100), unsafe_allow_html=True)

    scenarios = compute_whatif(inputs, res["prob"])

    if not scenarios:
        st.info("Your profile is already at maximum values for all scenarios! 🏆")
    else:
        # ── Delta bar chart ───────────────────────────────────────────────────
        labels_wi = [s[0] for s in scenarios]
        deltas    = [s[1] * 100 for s in scenarios]
        new_probs = [s[2] * 100 for s in scenarios]

        fig_wi, ax_wi = plt.subplots(figsize=(9, max(3, len(scenarios) * 0.75)))
        fig_wi.patch.set_facecolor("#13161e")
        ax_wi.set_facecolor("#0d0f14")

        bar_colors = ["#22c55e" if d > 0 else "#ef4444" for d in deltas]
        bars_wi = ax_wi.barh(labels_wi, deltas, color=bar_colors, height=0.5,
                             edgecolor="#0d0f14", zorder=3)

        for bar, delta, new_p in zip(bars_wi, deltas, new_probs):
            sign = "+" if delta >= 0 else ""
            ax_wi.text(
                delta + (0.2 if delta >= 0 else -0.2),
                bar.get_y() + bar.get_height()/2,
                f"{sign}{delta:.1f}pp → {new_p:.1f}%",
                va="center", ha="left" if delta >= 0 else "right",
                color="white", fontsize=8.5, fontweight="bold"
            )

        ax_wi.axvline(0, color="#374151", linewidth=1.2, zorder=2)
        ax_wi.set_xlabel("Change in Placement Probability (percentage points)", color="#6b7280", fontsize=9)
        ax_wi.tick_params(colors="#9ca3af", labelsize=9)
        ax_wi.spines[:].set_color("#1f2330")
        ax_wi.grid(axis="x", color="#1f2330", linewidth=0.5, zorder=0)
        ax_wi.set_title("Impact of Each Improvement on Placement Probability",
                        color="#9ca3af", fontsize=10, pad=10)
        plt.tight_layout()
        st.pyplot(fig_wi, use_container_width=True)
        plt.close(fig_wi)

        st.markdown("<div class='section-head'>Scenario Breakdown</div>", unsafe_allow_html=True)

        for label, delta, new_prob in scenarios:
            delta_pct = delta * 100
            sign = "+" if delta_pct >= 0 else ""
            color = "#22c55e" if delta_pct > 0.5 else ("#f59e0b" if delta_pct > 0 else "#ef4444")
            st.markdown(f"""
            <div class="whatif-card">
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span style='font-size:0.95rem;color:#c7caff;font-weight:600;'>{label}</span>
                <span style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:{color};'>
                  {sign}{delta_pct:.1f}pp
                </span>
              </div>
              <div style='font-size:0.82rem;color:#6b7280;margin-top:0.3rem;'>
                New placement probability: <b style='color:#9ca3af;'>{new_prob*100:.1f}%</b>
              </div>
              <div class='conf-bar-bg' style='margin-top:0.5rem;'>
                <div class='conf-bar-fill' style='width:{new_prob*100:.1f}%;background:{color};'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style='font-size:0.78rem;color:#374151;text-align:center;margin-top:1rem;'>
          💡 Tip: Combine multiple improvements for compounding effects!
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# TAB 4 — Dataset Explorer
# ============================================================================
with tab4:
    if DATASET is None:
        st.warning("Dataset CSV not found. Place `student_placement_salary_elite_v2.csv` in the project root.")
    else:
        df = DATASET.copy()

        # ── KPI row ──────────────────────────────────────────────────────────
        st.markdown("<div class='section-head'>Dataset Overview</div>", unsafe_allow_html=True)
        k1, k2, k3, k4, k5 = st.columns(5)
        for col, num, label in zip(
            [k1, k2, k3, k4, k5],
            [
                f"{len(df):,}",
                f"{df['placed'].mean()*100:.1f}%",
                f"₹{df['salary_lpa'].median():.1f}",
                f"{df['cgpa'].mean():.2f}",
                f"{df['internships'].mean():.2f}",
            ],
            ["Total Students", "Placement Rate", "Median Salary", "Avg CGPA", "Avg Internships"]
        ):
            with col:
                st.markdown(f"""
                <div class="metric-pill">
                  <div class="num" style='font-size:1.3rem;'>{num}</div>
                  <div class="lbl">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Row 1: Placement by Tier + Branch ────────────────────────────────
        st.markdown("<div class='section-head'>Placement Rates by College Tier & Branch</div>",
                    unsafe_allow_html=True)
        dc1, dc2 = st.columns(2)

        with dc1:
            tier_grp = df.groupby("college_tier")["placed"].mean() * 100
            fig_t, ax_t = plt.subplots(figsize=(5, 3.5))
            fig_t.patch.set_facecolor("#13161e"); ax_t.set_facecolor("#0d0f14")
            bars_t = ax_t.bar(
                [f"Tier {t}" for t in tier_grp.index], tier_grp.values,
                color=["#7c83f5","#5b63e8","#3a42d0"], edgecolor="#0d0f14", zorder=3
            )
            for b, v in zip(bars_t, tier_grp.values):
                ax_t.text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.1f}%",
                          ha="center", color="white", fontsize=9, fontweight="bold")
            ax_t.set_ylim(0, 105); ax_t.set_ylabel("Placement %", color="#6b7280", fontsize=9)
            ax_t.tick_params(colors="#9ca3af"); ax_t.spines[:].set_color("#1f2330")
            ax_t.grid(axis="y", color="#1f2330", linewidth=0.5, zorder=0)
            ax_t.set_title("Placement Rate by College Tier", color="#9ca3af", fontsize=10, pad=8)
            plt.tight_layout(); st.pyplot(fig_t, use_container_width=True); plt.close(fig_t)

        with dc2:
            branch_grp = df.groupby("branch")["placed"].mean().sort_values(ascending=False) * 100
            fig_b, ax_b = plt.subplots(figsize=(5, 3.5))
            fig_b.patch.set_facecolor("#13161e"); ax_b.set_facecolor("#0d0f14")
            colors_b = ["#7c83f5" if b == branch else "#2a2f45" for b in branch_grp.index]
            bars_b = ax_b.bar(branch_grp.index, branch_grp.values,
                              color=colors_b, edgecolor="#0d0f14", zorder=3)
            for b_bar, v in zip(bars_b, branch_grp.values):
                ax_b.text(b_bar.get_x()+b_bar.get_width()/2, v+0.5, f"{v:.0f}%",
                          ha="center", color="white", fontsize=8.5, fontweight="bold")
            ax_b.set_ylim(0, 105); ax_b.set_ylabel("Placement %", color="#6b7280", fontsize=9)
            ax_b.tick_params(colors="#9ca3af", labelsize=8.5); ax_b.spines[:].set_color("#1f2330")
            ax_b.grid(axis="y", color="#1f2330", linewidth=0.5, zorder=0)
            ax_b.set_title("Placement by Branch (your branch highlighted)", color="#9ca3af", fontsize=10, pad=8)
            plt.tight_layout(); st.pyplot(fig_b, use_container_width=True); plt.close(fig_b)

        # ── Row 2: Salary distribution + Job role pie ─────────────────────────
        st.markdown("<div class='section-head'>Salary Distribution & Job Role Split</div>",
                    unsafe_allow_html=True)
        dc3, dc4 = st.columns(2)

        with dc3:
            fig_s, ax_s = plt.subplots(figsize=(5, 3.5))
            fig_s.patch.set_facecolor("#13161e"); ax_s.set_facecolor("#0d0f14")
            ax_s.hist(df["salary_lpa"], bins=30, color="#7c83f5", edgecolor="#0d0f14",
                      alpha=0.85, zorder=3)
            ax_s.axvline(res["salary"], color="#f59e0b", linewidth=2.2,
                         linestyle="--", label=f"You: ₹{res['salary']:.1f}L", zorder=4)
            ax_s.axvline(df["salary_lpa"].median(), color="#22c55e", linewidth=1.5,
                         linestyle=":", label=f"Median: ₹{df['salary_lpa'].median():.1f}L", zorder=4)
            ax_s.set_xlabel("Salary (LPA)", color="#6b7280", fontsize=9)
            ax_s.set_ylabel("Students", color="#6b7280", fontsize=9)
            ax_s.tick_params(colors="#9ca3af"); ax_s.spines[:].set_color("#1f2330")
            ax_s.grid(axis="y", color="#1f2330", linewidth=0.5, zorder=0)
            ax_s.legend(fontsize=8.5, labelcolor="#9ca3af", facecolor="#13161e", edgecolor="#1f2330")
            ax_s.set_title("Salary Distribution (9,000 students)", color="#9ca3af", fontsize=10, pad=8)
            plt.tight_layout(); st.pyplot(fig_s, use_container_width=True); plt.close(fig_s)

        with dc4:
            role_counts = df["job_role"].value_counts()
            fig_p, ax_p = plt.subplots(figsize=(5, 3.5))
            fig_p.patch.set_facecolor("#13161e"); ax_p.set_facecolor("#0d0f14")
            role_palette = ["#7c83f5","#22c55e","#f59e0b","#ef4444"]
            explode = [0.04 if r == res["role"] else 0 for r in role_counts.index]
            wedges, texts, autotexts = ax_p.pie(
                role_counts.values, labels=role_counts.index,
                colors=role_palette, autopct="%1.0f%%",
                startangle=140, explode=explode,
                textprops={"color":"#9ca3af","fontsize":9},
                wedgeprops={"edgecolor":"#0d0f14","linewidth":1.5}
            )
            for at in autotexts: at.set_color("white"); at.set_fontweight("bold")
            ax_p.set_title(f"Job Role Distribution\n(your prediction: {res['role']} highlighted)",
                           color="#9ca3af", fontsize=9.5, pad=8)
            plt.tight_layout(); st.pyplot(fig_p, use_container_width=True); plt.close(fig_p)

        # ── Row 3: CGPA vs Placement + Skill Adoption ─────────────────────────
        st.markdown("<div class='section-head'>CGPA vs Placement & Skill Adoption Rates</div>",
                    unsafe_allow_html=True)
        dc5, dc6 = st.columns(2)

        with dc5:
            cgpa_bins = pd.cut(df["cgpa"], bins=[5,6,7,8,9,10], labels=["5-6","6-7","7-8","8-9","9-10"])
            cgpa_place = df.groupby(cgpa_bins, observed=True)["placed"].mean() * 100
            fig_c, ax_c = plt.subplots(figsize=(5, 3.5))
            fig_c.patch.set_facecolor("#13161e"); ax_c.set_facecolor("#0d0f14")
            ax_c.plot(cgpa_place.index, cgpa_place.values, color="#7c83f5",
                      marker="o", linewidth=2.5, markersize=7, zorder=3)
            ax_c.fill_between(range(len(cgpa_place)), cgpa_place.values,
                              color="#7c83f5", alpha=0.15)
            ax_c.set_xticks(range(len(cgpa_place)))
            ax_c.set_xticklabels(cgpa_place.index, color="#9ca3af", fontsize=9)
            ax_c.set_ylabel("Placement %", color="#6b7280", fontsize=9)
            ax_c.tick_params(colors="#9ca3af"); ax_c.spines[:].set_color("#1f2330")
            ax_c.grid(color="#1f2330", linewidth=0.5, zorder=0)
            ax_c.set_title("Placement Rate by CGPA Band", color="#9ca3af", fontsize=10, pad=8)
            plt.tight_layout(); st.pyplot(fig_c, use_container_width=True); plt.close(fig_c)

        with dc6:
            skill_cols = {"Python":"python_skill","DSA":"dsa_skill",
                          "ML":"ml_skill","Web Dev":"web_dev_skill"}
            skill_rates = {k: df[v].mean()*100 for k, v in skill_cols.items()}
            student_skill_rates = {
                "Python": int(python)*100, "DSA": int(dsa)*100,
                "ML": int(ml)*100, "Web Dev": int(webdev)*100
            }
            fig_sk, ax_sk = plt.subplots(figsize=(5, 3.5))
            fig_sk.patch.set_facecolor("#13161e"); ax_sk.set_facecolor("#0d0f14")
            xsk = np.arange(4)
            ax_sk.bar(xsk - 0.2, list(skill_rates.values()), 0.35,
                      color="#2a2f45", label="Dataset Avg", zorder=3)
            ax_sk.bar(xsk + 0.2, list(student_skill_rates.values()), 0.35,
                      color="#7c83f5", label="You", zorder=3)
            ax_sk.set_xticks(xsk)
            ax_sk.set_xticklabels(list(skill_cols.keys()), color="#9ca3af", fontsize=9)
            ax_sk.set_ylabel("% with Skill", color="#6b7280", fontsize=9)
            ax_sk.set_ylim(0, 115)
            ax_sk.tick_params(colors="#9ca3af"); ax_sk.spines[:].set_color("#1f2330")
            ax_sk.grid(axis="y", color="#1f2330", linewidth=0.5, zorder=0)
            ax_sk.legend(fontsize=9, labelcolor="#9ca3af", facecolor="#13161e", edgecolor="#1f2330")
            ax_sk.set_title("Skill Adoption Rate: You vs Dataset", color="#9ca3af", fontsize=10, pad=8)
            plt.tight_layout(); st.pyplot(fig_sk, use_container_width=True); plt.close(fig_sk)


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#374151;font-size:0.78rem;margin-top:2rem;padding-top:1rem;border-top:1px solid #1f2330;'>
  Software Career Predictor · Trained on 9,000 students · 3 ML models · Built with Streamlit
  &nbsp;·&nbsp; <a href='https://github.com/prashantsaha18/placement-predictor' style='color:#7c83f5;text-decoration:none;'>GitHub</a>
</div>
""", unsafe_allow_html=True)
