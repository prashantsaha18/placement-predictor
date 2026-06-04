"""
utils/helpers.py
Shared constants, CSS injection, model loader, and card-builder helpers
for the Software Career Predictor multi-page app.
"""

import pathlib
import joblib
import streamlit as st
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ── Constants ─────────────────────────────────────────────────────────────────
JOB_ROLES  = sorted(["Analyst", "Data Scientist", "Software Engineer", "Web Developer"])
ROLE_ICONS = {
    "Analyst":           "📊",
    "Data Scientist":    "🧠",
    "Software Engineer": "💻",
    "Web Developer":     "🌐",
}
ROLE_COLORS = {
    "Analyst":           "#7c83f5",
    "Data Scientist":    "#22c55e",
    "Software Engineer": "#f59e0b",
    "Web Developer":     "#ef4444",
}
BRANCHES = ["CSE", "ECE", "IT", "EEE", "Civil", "Mechanical"]

NUMERIC_COLS = [
    "cgpa", "python_skill", "dsa_skill", "ml_skill", "web_dev_skill",
    "coding_score", "communication_score", "aptitude_score",
    "internships", "projects", "backlogs", "resume_score", "skill_score",
]
CATEGORICAL_COLS = ["branch", "college_tier"]

# Dataset-derived averages (from training run)
DATASET_AVGS = {
    "placement_rate": 0.856,
    "avg_cgpa":       7.51,
    "median_salary":  65.9,
    "python_rate":    0.65,
    "dsa_rate":       0.56,
    "ml_rate":        0.30,
    "webdev_rate":    0.39,
    "avg_internships":0.98,
    "avg_projects":   3.12,
}

SKILL_P25  = 49.0
SKILL_P50  = 65.9
SKILL_P75  = 77.5
SKILL_P90  = 90.0


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    base = ROOT / "models"
    return {
        "placement": joblib.load(base / "placement_model.pkl"),
        "job_role":  joblib.load(base / "job_role_model.pkl"),
        "salary":    joblib.load(base / "salary_model.pkl"),
        "encoder":   joblib.load(base / "encoder.pkl"),
    }


@st.cache_data(show_spinner=False, ttl=3600)
def load_dataset():
    csv_path = ROOT / "student_placement_salary_elite_v2.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(inputs: dict, models: dict) -> dict:
    row = {c: [inputs[c]] for c in NUMERIC_COLS + CATEGORICAL_COLS}
    df  = pd.DataFrame(row)
    X   = models["encoder"].transform(df)

    place_prob = float(models["placement"].predict_proba(X)[0, 1])
    placed     = place_prob >= 0.5

    role_idx   = int(models["job_role"].predict(X)[0])
    role_name  = JOB_ROLES[role_idx]
    role_probas = {}
    if hasattr(models["job_role"], "predict_proba"):
        rp = models["job_role"].predict_proba(X)[0]
        role_probas = {JOB_ROLES[i]: float(p) for i, p in enumerate(rp)}

    salary = float(models["salary"].predict(X)[0])

    # Feature importances for explainability
    explanation = {}
    try:
        # Logistic regression coefficients for placement
        coef = models["placement"].coef_[0]
        feat_names = (models["encoder"]
                      .get_feature_names_out(NUMERIC_COLS + CATEGORICAL_COLS)
                      if hasattr(models["encoder"], "get_feature_names_out")
                      else NUMERIC_COLS)
        xarr = X[0] if hasattr(X, "__getitem__") else X.toarray()[0]
        contribs = {str(feat_names[i]): float(coef[i] * xarr[i])
                    for i in range(min(len(feat_names), len(coef)))}
        explanation["placement_contribs"] = contribs
    except Exception:
        pass

    try:
        importances = models["salary"].feature_importances_
        feat_names2 = (models["encoder"]
                       .get_feature_names_out(NUMERIC_COLS + CATEGORICAL_COLS)
                       if hasattr(models["encoder"], "get_feature_names_out")
                       else NUMERIC_COLS)
        explanation["salary_importances"] = {
            str(feat_names2[i]): float(importances[i])
            for i in range(min(len(feat_names2), len(importances)))
        }
    except Exception:
        pass

    return {
        "placed":       placed,
        "prob":         place_prob,
        "role":         role_name,
        "role_probas":  role_probas,
        "salary":       max(0.0, salary),
        "explanation":  explanation,
    }


# ── Readiness score ───────────────────────────────────────────────────────────
def compute_readiness(inputs: dict, prob: float) -> dict:
    cgpa_norm  = (inputs["cgpa"] - 5.0) / 5.0 * 25
    prob_norm  = prob * 30
    skill_norm = inputs["skill_score"] / 4 * 15
    exp_norm   = min(inputs["internships"] / 3, 1) * 15
    score_norm = (inputs["coding_score"] / 100 * 5 +
                  inputs["resume_score"] / 100 * 5 +
                  (inputs["aptitude_score"] - 40) / 60 * 5)
    total = min(100.0, max(0.0, cgpa_norm + prob_norm + skill_norm + exp_norm + score_norm))

    try:
        from scipy.stats import norm
        pct = float(norm.cdf(total, loc=62, scale=14)) * 100
    except ImportError:
        pct = total  # fallback

    if total >= 85:
        tier, color, glow = "Platinum", "#e2e8f0", "platinum"
    elif total >= 70:
        tier, color, glow = "Gold",     "#f59e0b", "gold"
    elif total >= 55:
        tier, color, glow = "Silver",   "#94a3b8", "silver"
    else:
        tier, color, glow = "Bronze",   "#b45309", "bronze"

    return {"score": total, "pct": pct, "tier": tier, "color": color, "glow": glow}


# ── Advisor tips ──────────────────────────────────────────────────────────────
def get_advisor_tips(inputs: dict, res: dict) -> list:
    tips = []
    if res["prob"] >= 0.85:
        tips.insert(0, ("🌟", "You're in Great Shape!",
                         f"Placement probability {res['prob']*100:.1f}%. "
                         "Focus on dream companies, early applications, and salary negotiation."))
    if inputs["cgpa"] < 7.0:
        tips.append(("📚", "Improve Your CGPA",
                     f"Your CGPA of {inputs['cgpa']:.1f} is below the 7.0 cutoff many companies enforce. "
                     "Focus on core subjects this semester."))
    if inputs["internships"] == 0:
        tips.append(("🏢", "Get an Internship",
                     "No internship experience yet — the #1 placement differentiator. "
                     "Apply to FAANG internships, startups, and research labs immediately."))
    elif inputs["internships"] == 1:
        tips.append(("🏢", "Add a 2nd Internship",
                     "Students with 2+ internships have 18% higher placement rates. "
                     "Seek part-time or remote roles."))
    if inputs["dsa_skill"] == 0:
        tips.append(("🧩", "Learn DSA",
                     "DSA is tested in virtually every technical interview. "
                     "Practice 2–3 LeetCode problems daily for 60 days."))
    if inputs["coding_score"] < 55:
        tips.append(("💻", "Boost Coding Score",
                     f"Score {inputs['coding_score']} — below median. "
                     "Join Codeforces / HackerRank weekly contests."))
    if inputs["resume_score"] < 60:
        tips.append(("📄", "Strengthen Resume",
                     f"Resume score {inputs['resume_score']}. "
                     "Use a single-page format, quantify achievements, get r/cscareerquestions feedback."))
    if inputs["projects"] < 3:
        tips.append(("🚀", "Build More Projects",
                     f"Only {inputs['projects']} project(s). Aim for 3–4: "
                     "deployed web app, ML pipeline, DSA-focused GitHub repo."))
    if inputs["backlogs"] > 0:
        tips.append(("⚠️", "Clear Backlogs",
                     f"{inputs['backlogs']} backlog(s) flagged in 80% of background checks. "
                     "Prioritize before campus placements open."))
    return tips[:5]


# ── What-If scenarios ─────────────────────────────────────────────────────────
def compute_whatif(base_inputs: dict, base_prob: float, models: dict) -> list:
    scenarios = []

    def _try(label, mod):
        inp2 = {**base_inputs, **mod}
        r2   = predict(inp2, models)
        scenarios.append((label, r2["prob"] - base_prob, r2["prob"]))

    if base_inputs["cgpa"] < 10.0:
        _try("📈 CGPA +0.5", {"cgpa": min(10.0, base_inputs["cgpa"] + 0.5)})
    if base_inputs["internships"] < 3:
        _try("🏢 +1 Internship", {"internships": base_inputs["internships"] + 1})
    if base_inputs["projects"] < 6:
        _try("🚀 +1 Project", {"projects": base_inputs["projects"] + 1})
    if base_inputs["dsa_skill"] == 0:
        _try("🧩 Learn DSA", {"dsa_skill": 1, "skill_score": min(4, base_inputs["skill_score"] + 1)})
    if base_inputs["python_skill"] == 0:
        _try("🐍 Learn Python", {"python_skill": 1, "skill_score": min(4, base_inputs["skill_score"] + 1)})
    if base_inputs["ml_skill"] == 0:
        _try("🤖 Learn ML", {"ml_skill": 1, "skill_score": min(4, base_inputs["skill_score"] + 1)})
    if base_inputs["coding_score"] <= 85:
        _try("💻 Coding +15", {"coding_score": min(100, base_inputs["coding_score"] + 15)})
    if base_inputs["resume_score"] <= 80:
        _try("📄 Resume +10", {"resume_score": min(100, base_inputs["resume_score"] + 10)})
    if base_inputs["backlogs"] > 0:
        _try("✅ Clear Backlogs", {"backlogs": 0})
    if base_inputs["aptitude_score"] <= 85:
        _try("🧮 Aptitude +10", {"aptitude_score": min(100, base_inputs["aptitude_score"] + 10)})

    scenarios.sort(key=lambda x: -x[1])
    return scenarios


# ── CSS injector ──────────────────────────────────────────────────────────────
SHARED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Background & base ── */
.stApp { background: #080a0f; color: #e8eaf0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1117 0%, #13161e 100%);
    border-right: 1px solid #1f2330;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #7c83f5; font-family: 'Syne', sans-serif;
    font-weight: 700; letter-spacing: 0.03em;
}
[data-testid="stSidebarNav"] { padding-top: 0.5rem; }
[data-testid="stSidebarNavLink"] {
    border-radius: 10px; margin: 2px 8px;
    transition: background 0.2s;
}
[data-testid="stSidebarNavLink"]:hover { background: #1f2330; }
[data-testid="stSidebarNavLink"][aria-current="page"] {
    background: linear-gradient(135deg, #1f2a4a, #1a1f35) !important;
    border-left: 3px solid #7c83f5;
}

/* ── Cards ── */
.card {
    background: #13161e;
    border: 1px solid #1f2330;
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1rem;
    transition: transform 0.2s, box-shadow 0.2s;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(124,131,245,0.12);
}
.card-placed    { border-left: 4px solid #22c55e; }
.card-rejected  { border-left: 4px solid #ef4444; }
.card-role      { border-left: 4px solid #7c83f5; }
.card-salary    { border-left: 4px solid #f59e0b; }
.card-info      { border-left: 4px solid #38bdf8; }

.card-label {
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #4b5563; margin-bottom: 0.4rem;
}
.card-value {
    font-family: 'Syne', sans-serif; font-size: 1.9rem;
    font-weight: 800; color: #fff; line-height: 1.1;
}
.card-sub { font-size: 0.83rem; color: #9ca3af; margin-top: 0.3rem; }

/* ── Section headers ── */
.section-head {
    font-family: 'Syne', sans-serif; font-size: 1rem;
    font-weight: 700; color: #c7caff; letter-spacing: 0.06em;
    text-transform: uppercase; border-bottom: 1px solid #1f2330;
    padding-bottom: 0.5rem; margin: 1.8rem 0 1rem 0;
}

/* ── Confidence bar ── */
.bar-bg {
    background: #1f2330; border-radius: 99px;
    height: 6px; width: 100%; overflow: hidden; margin-top: 0.6rem;
}
.bar-fill { height: 100%; border-radius: 99px; transition: width 0.6s ease; }

/* ── Metric pill ── */
.metric-pill {
    background: #13161e; border: 1px solid #1f2330;
    border-radius: 10px; padding: 1rem 1.2rem; text-align: center;
    transition: transform 0.2s;
}
.metric-pill:hover { transform: translateY(-2px); }
.metric-pill .num {
    font-family: 'Syne', sans-serif; font-size: 1.5rem;
    font-weight: 800; color: #7c83f5;
}
.metric-pill .lbl {
    font-size: 0.68rem; color: #4b5563;
    text-transform: uppercase; letter-spacing: 0.1em;
}

/* ── Chips ── */
.chip {
    display: inline-block; background: #1a1d2e;
    border: 1px solid #2a2f45; border-radius: 99px;
    padding: 0.2rem 0.75rem; font-size: 0.78rem;
    color: #a5b4fc; margin: 0.2rem;
}

/* ── Advisor card ── */
.advisor-card {
    background: linear-gradient(135deg, #13161e, #1a1d2e);
    border: 1px solid #2a2f45; border-radius: 12px;
    padding: 1rem 1.4rem; margin-bottom: 0.7rem;
    display: flex; align-items: flex-start; gap: 0.8rem;
    animation: fadeSlideIn 0.4s ease both;
}
.advisor-icon { font-size: 1.4rem; flex-shrink: 0; margin-top: 0.1rem; }
.advisor-title {
    font-family: 'Syne', sans-serif; font-size: 0.9rem;
    font-weight: 700; color: #c7caff; margin-bottom: 0.2rem;
}
.advisor-body { font-size: 0.82rem; color: #9ca3af; line-height: 1.5; }

/* ── Badge ── */
.badge-wrap {
    display: flex; flex-direction: column; align-items: center;
    padding: 1.5rem; background: #13161e; border-radius: 14px;
    border: 1px solid #1f2330; margin-bottom: 1rem;
}
.badge-circle {
    width: 110px; height: 110px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-direction: column; font-family: 'Syne', sans-serif;
    margin-bottom: 0.8rem;
}
.badge-score { font-size: 1.8rem; font-weight: 800; color: white; line-height: 1; }
.badge-lbl { font-size: 0.62rem; letter-spacing: 0.12em; color: rgba(255,255,255,0.6); }
.badge-tier {
    font-family: 'Syne', sans-serif; font-size: 1rem;
    font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
}
.badge-pct { font-size: 0.8rem; color: #4b5563; margin-top: 0.25rem; }

/* ── Roadmap timeline ── */
.timeline-item {
    display: flex; gap: 1rem; margin-bottom: 1.2rem; align-items: flex-start;
}
.timeline-dot {
    width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0;
    margin-top: 5px; box-shadow: 0 0 8px currentColor;
}
.timeline-line {
    width: 2px; background: #1f2330; margin: 4px 0 0 5px; flex-shrink: 0;
}
.timeline-content {
    background: #13161e; border: 1px solid #1f2330;
    border-radius: 10px; padding: 0.9rem 1.2rem; flex: 1;
}
.timeline-week {
    font-family: 'Syne', sans-serif; font-size: 0.7rem;
    font-weight: 700; letter-spacing: 0.1em; color: #4b5563;
    text-transform: uppercase; margin-bottom: 0.3rem;
}
.timeline-title {
    font-family: 'Syne', sans-serif; font-size: 0.95rem;
    font-weight: 700; color: #c7caff; margin-bottom: 0.3rem;
}
.timeline-body { font-size: 0.82rem; color: #6b7280; line-height: 1.55; }

/* ── Trend card ── */
.trend-card {
    background: linear-gradient(135deg, #13161e, #111420);
    border: 1px solid #1f2330; border-radius: 12px;
    padding: 1.2rem 1.5rem; margin-bottom: 0.8rem;
    transition: transform 0.2s, box-shadow 0.2s;
}
.trend-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(124,131,245,0.1);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c83f5, #5b63e8);
    color: white; border: none; border-radius: 10px;
    font-family: 'Syne', sans-serif; font-weight: 700;
    font-size: 0.95rem; letter-spacing: 0.04em;
    padding: 0.65rem 1.4rem; width: 100%; transition: all 0.2s;
}
.stButton > button:hover {
    opacity: 0.9; transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(124,131,245,0.4);
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #0a2a0a, #0d350d);
    border: 1px solid #22c55e; color: #22c55e !important;
}
.stDownloadButton > button:hover { opacity: 0.85; }

/* ── Slider / labels ── */
label { color: #9ca3af !important; font-size: 0.85rem !important; }
.stSlider > div > div > div > div { background: #7c83f5 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #13161e; border-radius: 10px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 8px; color: #6b7280;
    font-family: 'Syne', sans-serif; font-weight: 600; font-size: 0.88rem;
}
.stTabs [aria-selected="true"] {
    background: #1f2330 !important; color: #c7caff !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div { background: #7c83f5 !important; }

/* ── Animations ── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; } to { opacity: 1; }
}
@keyframes glowPulse {
    0%,100% { box-shadow: 0 0 18px rgba(124,131,245,0.3); }
    50%      { box-shadow: 0 0 36px rgba(124,131,245,0.65); }
}
@keyframes countUp {
    from { opacity: 0; transform: scale(0.8); }
    to   { opacity: 1; transform: scale(1); }
}
.badge-circle.platinum { animation: glowPulse 2.5s ease-in-out infinite; }
.animate-in { animation: fadeSlideIn 0.45s ease both; }

/* ── Divider ── */
hr { border-color: #1f2330; margin: 1.2rem 0; }

/* ── Input overrides ── */
[data-testid="stSelectbox"] > div > div { background: #13161e; border-color: #1f2330; }
[data-testid="stNumberInput"] input { background: #13161e; border-color: #1f2330; color: #e8eaf0; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #13161e; border-radius: 8px; color: #c7caff !important;
    font-family: 'Syne', sans-serif; font-weight: 600;
}

/* ── Alert / info boxes ── */
.stAlert { border-radius: 10px; }
</style>
"""


def inject_css():
    st.markdown(SHARED_CSS, unsafe_allow_html=True)


# ── HTML helpers ──────────────────────────────────────────────────────────────
def result_card(label: str, value: str, sub: str, cls: str, bar_pct: float = None, bar_color: str = "#7c83f5") -> str:
    bar_html = ""
    if bar_pct is not None:
        bar_html = f"""
        <div class="bar-bg">
          <div class="bar-fill" style="width:{bar_pct:.1f}%;background:{bar_color};"></div>
        </div>"""
    return f"""
    <div class="card {cls}">
      <div class="card-label">{label}</div>
      <div class="card-value">{value}</div>
      <div class="card-sub">{sub}</div>
      {bar_html}
    </div>"""


def section_head(title: str) -> str:
    return f"<div class='section-head'>{title}</div>"


def metric_pill(num: str, label: str) -> str:
    return f"""
    <div class="metric-pill">
      <div class="num">{num}</div>
      <div class="lbl">{label}</div>
    </div>"""


def advisor_card(icon: str, title: str, body: str) -> str:
    return f"""
    <div class="advisor-card">
      <div class="advisor-icon">{icon}</div>
      <div>
        <div class="advisor-title">{title}</div>
        <div class="advisor-body">{body}</div>
      </div>
    </div>"""


def badge_html(score: float, pct: float, tier: str, color: str, glow: str) -> str:
    tier_bg = {
        "Platinum": "linear-gradient(135deg,#334155,#64748b)",
        "Gold":     "linear-gradient(135deg,#78350f,#d97706)",
        "Silver":   "linear-gradient(135deg,#334155,#64748b)",
        "Bronze":   "linear-gradient(135deg,#451a03,#92400e)",
    }.get(tier, "linear-gradient(135deg,#1e293b,#334155)")
    return f"""
    <div class="badge-wrap">
      <div class="badge-circle {glow}" style="background:{tier_bg};">
        <div class="badge-score">{score:.0f}</div>
        <div class="badge-lbl">/ 100</div>
      </div>
      <div class="badge-tier" style="color:{color};">{tier} Tier</div>
      <div class="badge-pct">Top {100-pct:.0f}% of students</div>
    </div>"""


def role_prob_bars(role_probas: dict, predicted_role: str) -> str:
    html = ""
    for role, prob in sorted(role_probas.items(), key=lambda x: -x[1]):
        pct    = prob * 100
        is_top = role == predicted_role
        bar_c  = ROLE_COLORS.get(role, "#7c83f5") if is_top else "#2a2f45"
        icon   = ROLE_ICONS.get(role, "💼")
        html += f"""
        <div style="margin-bottom:0.55rem;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:0.88rem;color:{'#c7caff' if is_top else '#6b7280'};">
              {icon} {role}{'  ◀' if is_top else ''}
            </span>
            <span style="font-size:0.83rem;color:#9ca3af;">{pct:.1f}%</span>
          </div>
          <div class="bar-bg">
            <div class="bar-fill" style="width:{pct:.1f}%;background:{bar_c};"></div>
          </div>
        </div>"""
    return html


def sidebar_profile_form():
    """Render the sidebar form and return (inputs dict, predict_btn)."""
    with st.sidebar:
        st.markdown("## 🎯 Student Profile")
        st.markdown("---")

        st.markdown("### 🏫 Academic")
        cgpa     = st.slider("CGPA", 5.0, 10.0, 7.8, 0.1, key="cgpa")
        branch   = st.selectbox("Branch", BRANCHES, key="branch")
        tier     = st.selectbox("College Tier", [1, 2, 3],
                                format_func=lambda x: f"Tier {x}", key="tier")
        backlogs = st.slider("Backlogs", 0, 3, 0, key="backlogs")

        st.markdown("### 🛠️ Technical Skills")
        python = st.toggle("Python",           value=True,  key="python")
        dsa    = st.toggle("DSA",              value=True,  key="dsa")
        ml     = st.toggle("Machine Learning", value=False, key="ml")
        webdev = st.toggle("Web Development",  value=False, key="webdev")

        st.markdown("### 📊 Scores")
        coding = st.slider("Coding Score",        0,   100, 72,  key="coding")
        comm   = st.slider("Communication Score", 4.0, 10.0, 7.0, 0.5, key="comm")
        apt    = st.slider("Aptitude Score",      40,  100, 75,  key="apt")
        resume = st.slider("Resume Score",        0,   100, 70,  key="resume")

        st.markdown("### 💼 Experience")
        internships = st.selectbox("Internships", [0,1,2,3],      key="internships")
        projects    = st.selectbox("Projects",    [1,2,3,4,5,6],  key="projects")

        skill_score = int(python) + int(dsa) + int(ml) + int(webdev)

        st.markdown("---")
        predict_btn = st.button("🔮 Predict", use_container_width=True, key="predict_btn")

    inp = dict(
        cgpa=cgpa, python_skill=int(python), dsa_skill=int(dsa),
        ml_skill=int(ml), web_dev_skill=int(webdev),
        coding_score=float(coding), communication_score=float(comm),
        aptitude_score=float(apt), internships=internships,
        projects=projects, backlogs=backlogs,
        resume_score=float(resume), skill_score=skill_score,
        branch=branch, college_tier=str(tier),
    )
    return inp, predict_btn
