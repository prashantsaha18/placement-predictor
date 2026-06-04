"""
app.py — Premium Landing Page
Software Career Predictor
"""
import sys
import pathlib
import warnings
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import streamlit as st
import time

st.set_page_config(
    page_title="Software Career Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from utils.helpers import inject_css, DATASET_AVGS

inject_css()

# ── Extra landing-page CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hero */
.hero-wrap {
    background: linear-gradient(135deg, #0d1020 0%, #0a0e1a 45%, #060912 100%);
    border: 1px solid #1f2a40;
    border-radius: 20px;
    padding: 3.5rem 3.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    animation: fadeIn 0.6s ease both;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 360px; height: 360px;
    background: radial-gradient(circle, rgba(124,131,245,0.18) 0%, transparent 65%);
    pointer-events: none;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 30%;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(34,197,94,0.08) 0%, transparent 65%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: #7c83f5; margin-bottom: 0.9rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.hero-eyebrow::before {
    content: ''; display: inline-block;
    width: 28px; height: 2px; background: #7c83f5; border-radius: 2px;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem; font-weight: 800;
    color: #ffffff; line-height: 1.06;
    letter-spacing: -0.03em; margin-bottom: 1.2rem;
}
.hero-title span { color: #7c83f5; }
.hero-desc {
    font-size: 1.05rem; color: #6b7280;
    line-height: 1.7; max-width: 560px; margin-bottom: 2rem;
}
.hero-cta {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: linear-gradient(135deg, #7c83f5, #5b63e8);
    color: white; border: none; border-radius: 12px;
    padding: 0.85rem 2rem; font-family: 'Syne', sans-serif;
    font-weight: 700; font-size: 1rem; letter-spacing: 0.03em;
    text-decoration: none; cursor: pointer;
    transition: all 0.2s; box-shadow: 0 4px 20px rgba(124,131,245,0.4);
}
.hero-cta:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(124,131,245,0.55); }
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #0d1520; border: 1px solid #1f2a40;
    border-radius: 99px; padding: 0.3rem 0.9rem;
    font-size: 0.78rem; color: #4b5563; margin: 0 0.3rem 0.4rem 0;
}
.hero-badge span { color: #7c83f5; font-weight: 600; }

/* Stat counter */
.stat-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 1rem; margin-bottom: 2rem;
}
.stat-card {
    background: #13161e; border: 1px solid #1f2330;
    border-radius: 14px; padding: 1.4rem 1.5rem;
    text-align: center; position: relative; overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
    animation: fadeSlideIn 0.5s ease both;
}
.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(124,131,245,0.14);
}
.stat-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #7c83f5, #22c55e);
}
.stat-num {
    font-family: 'Syne', sans-serif; font-size: 2.4rem;
    font-weight: 800; color: #fff; line-height: 1;
    animation: countUp 0.6s ease both;
}
.stat-lbl { font-size: 0.75rem; color: #4b5563; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.4rem; }
.stat-icon { font-size: 1.4rem; margin-bottom: 0.5rem; }

/* Feature cards */
.feature-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 1rem; margin-bottom: 2rem;
}
.feature-card {
    background: linear-gradient(135deg, #13161e, #111420);
    border: 1px solid #1f2330; border-radius: 14px;
    padding: 1.6rem 1.8rem;
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    animation: fadeSlideIn 0.5s ease both;
    cursor: default;
}
.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(124,131,245,0.16);
    border-color: #3a3f5c;
}
.feature-icon { font-size: 2rem; margin-bottom: 0.9rem; }
.feature-title {
    font-family: 'Syne', sans-serif; font-size: 1rem;
    font-weight: 700; color: #c7caff; margin-bottom: 0.5rem;
}
.feature-desc { font-size: 0.83rem; color: #6b7280; line-height: 1.6; }

/* How it works */
.how-step {
    display: flex; gap: 1.2rem; align-items: flex-start;
    background: #13161e; border: 1px solid #1f2330;
    border-radius: 12px; padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem; transition: transform 0.2s;
}
.how-step:hover { transform: translateX(4px); }
.step-num {
    font-family: 'Syne', sans-serif; font-size: 1.4rem;
    font-weight: 800; color: #7c83f5; flex-shrink: 0;
    width: 36px; line-height: 1;
}
.step-title {
    font-family: 'Syne', sans-serif; font-size: 0.92rem;
    font-weight: 700; color: #c7caff; margin-bottom: 0.3rem;
}
.step-body { font-size: 0.82rem; color: #6b7280; line-height: 1.5; }

/* Tech badge */
.tech-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #13161e; border: 1px solid #1f2330;
    border-radius: 8px; padding: 0.45rem 0.9rem;
    font-size: 0.82rem; color: #9ca3af; margin: 0.25rem;
    transition: border-color 0.2s, color 0.2s;
}
.tech-pill:hover { border-color: #7c83f5; color: #c7caff; }

@media (max-width: 768px) {
    .stat-grid { grid-template-columns: repeat(2, 1fr); }
    .feature-grid { grid-template-columns: 1fr; }
    .hero-title { font-size: 2rem; }
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">AI-Powered Career Intelligence</div>
  <h1 class="hero-title">Predict Your<br><span>Software Career</span><br>With Confidence</h1>
  <p class="hero-desc">
    Enter your academic profile and get instant ML-powered predictions for
    placement probability, expected job role, and salary — backed by
    9,000 real student outcomes.
  </p>
  <div style="margin-bottom:1.5rem;">
    <span class="hero-badge">🎓 <span>9,000</span> Students Analyzed</span>
    <span class="hero-badge">🤖 <span>3</span> ML Models</span>
    <span class="hero-badge">🎯 <span>95.7%</span> Placement F1</span>
    <span class="hero-badge">📊 <span>4</span> Career Domains</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Navigate button
col_btn, col_blank = st.columns([1, 3])
with col_btn:
    if st.button("🚀 Launch Predictor →", use_container_width=True):
        st.switch_page("pages/1_🎯_Predictor.py")

st.markdown("<br>", unsafe_allow_html=True)

# ── Live Stats ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-icon">🎓</div>
    <div class="stat-num">9,000</div>
    <div class="stat-lbl">Students in Dataset</div>
  </div>
  <div class="stat-card">
    <div class="stat-icon">✅</div>
    <div class="stat-num">85.6%</div>
    <div class="stat-lbl">Overall Placement Rate</div>
  </div>
  <div class="stat-card">
    <div class="stat-icon">💰</div>
    <div class="stat-num">₹65.9L</div>
    <div class="stat-lbl">Median Salary (LPA)</div>
  </div>
  <div class="stat-card">
    <div class="stat-icon">🤖</div>
    <div class="stat-num">0.957</div>
    <div class="stat-lbl">Placement Model F1</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Features Grid ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-head'>✨ What You Get</div>", unsafe_allow_html=True)

st.markdown("""
<div class="feature-grid">
  <div class="feature-card">
    <div class="feature-icon">🎯</div>
    <div class="feature-title">Placement Predictor</div>
    <div class="feature-desc">
      Get your exact placement probability powered by a Logistic Regression model
      trained on 9,000 outcomes with 95.7% F1 score.
    </div>
  </div>
  <div class="feature-card">
    <div class="feature-icon">💼</div>
    <div class="feature-title">Job Role Prediction</div>
    <div class="feature-desc">
      Find out if you're best suited for Data Scientist, Software Engineer,
      Analyst, or Web Developer roles with probability breakdowns.
    </div>
  </div>
  <div class="feature-card">
    <div class="feature-icon">💰</div>
    <div class="feature-title">Salary Estimation</div>
    <div class="feature-desc">
      Random Forest salary model gives you expected LPA with confidence ranges
      and comparison against dataset percentiles.
    </div>
  </div>
  <div class="feature-card">
    <div class="feature-icon">🔍</div>
    <div class="feature-title">AI Explainability</div>
    <div class="feature-desc">
      SHAP-style feature contribution charts show exactly WHY the model
      predicted your outcome — full transparency.
    </div>
  </div>
  <div class="feature-card">
    <div class="feature-icon">🔄</div>
    <div class="feature-title">What-If Simulator</div>
    <div class="feature-desc">
      See the exact impact of gaining a skill, doing an internship, or
      clearing backlogs on your placement probability.
    </div>
  </div>
  <div class="feature-card">
    <div class="feature-icon">🗺️</div>
    <div class="feature-title">Career Roadmap</div>
    <div class="feature-desc">
      Get a personalized 90-day action plan with weekly milestones, resource
      links, and role-specific learning paths.
    </div>
  </div>
  <div class="feature-card">
    <div class="feature-icon">🏆</div>
    <div class="feature-title">Peer Comparison</div>
    <div class="feature-desc">
      Compare your profile against students with the same branch and tier —
      see exactly where you stand in your peer group.
    </div>
  </div>
  <div class="feature-card">
    <div class="feature-icon">📚</div>
    <div class="feature-title">Interview Prep Hub</div>
    <div class="feature-desc">
      Role-specific question banks, DSA topic checklists, and a readiness
      assessment to prepare you for campus placements.
    </div>
  </div>
  <div class="feature-card">
    <div class="feature-icon">📈</div>
    <div class="feature-title">Industry Trends</div>
    <div class="feature-desc">
      Explore salary trends by role, in-demand skills, top hiring companies,
      and remote vs on-site ratios across the industry.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── How It Works ──────────────────────────────────────────────────────────────
col_how, col_tech = st.columns([1.3, 1])

with col_how:
    st.markdown("<div class='section-head'>⚙️ How It Works</div>", unsafe_allow_html=True)
    steps = [
        ("01", "Enter Your Profile",
         "Fill in your CGPA, branch, skills, internships, projects, and assessment scores in the sidebar."),
        ("02", "ML Models Run Instantly",
         "Three models process your inputs: Logistic Regression (placement), Decision Tree (role), Random Forest (salary)."),
        ("03", "Get Comprehensive Insights",
         "View placement probability, predicted role, salary estimate, SHAP explanations, and percentile rank."),
        ("04", "Explore & Improve",
         "Use the What-If Simulator, Career Roadmap, and Peer Comparison to close your readiness gap."),
    ]
    for num, title, body in steps:
        st.markdown(f"""
        <div class="how-step">
          <div class="step-num">{num}</div>
          <div>
            <div class="step-title">{title}</div>
            <div class="step-body">{body}</div>
          </div>
        </div>""", unsafe_allow_html=True)

with col_tech:
    st.markdown("<div class='section-head'>🛠️ Tech Stack</div>", unsafe_allow_html=True)
    tech = [
        ("🐍","Python 3.9+"),("📊","scikit-learn"),("🌊","Streamlit"),
        ("🐼","Pandas"),("🔢","NumPy"),("📉","Matplotlib"),
        ("💾","Joblib"),("📐","SciPy"),("🤖","Logistic Regression"),
        ("🌲","Random Forest"),("🌳","Decision Tree"),("🔄","ColumnTransformer"),
    ]
    pills = "".join(f'<span class="tech-pill">{icon} {name}</span>' for icon, name in tech)
    st.markdown(f"<div style='margin-top:0.5rem;'>{pills}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-head' style='margin-top:2rem;'>📊 Model Performance</div>",
                unsafe_allow_html=True)
    perf = [
        ("Placement (LR)", "F1 Score", "0.957"),
        ("Job Role (DT)", "Accuracy", "25.2%*"),
        ("Salary (RF)", "R² Score", "0.622"),
    ]
    for model, metric, val in perf:
        st.markdown(f"""
        <div style="background:#13161e;border:1px solid #1f2330;border-radius:10px;
                    padding:0.8rem 1.2rem;margin-bottom:0.5rem;
                    display:flex;justify-content:space-between;align-items:center;">
          <div>
            <div style="font-size:0.88rem;color:#c7caff;font-weight:600;">{model}</div>
            <div style="font-size:0.72rem;color:#4b5563;">{metric}</div>
          </div>
          <div style="font-family:Syne,sans-serif;font-size:1.2rem;font-weight:800;color:#7c83f5;">{val}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem;color:#374151;margin-top:0.4rem;">
    *Job role assignment is random in synthetic dataset (4 equally balanced classes).
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="text-align:center;color:#374151;font-size:0.78rem;padding:0.5rem 0 1.5rem;">
  Software Career Predictor &nbsp;·&nbsp; Trained on 9,000 Students &nbsp;·&nbsp; 3 ML Models &nbsp;·&nbsp;
  Built with ❤️ using Streamlit &nbsp;·&nbsp;
  <a href="https://github.com/prashantsaha18/placement-predictor" style="color:#7c83f5;text-decoration:none;">GitHub</a>
</div>
""", unsafe_allow_html=True)
