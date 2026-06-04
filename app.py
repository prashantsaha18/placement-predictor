"""
app.py — Streamlit frontend for Software Career Predictor
Run: streamlit run app.py
"""

import sys
import pathlib
import warnings
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

# ── Run prediction on button click OR on first load ───────────────────────────
inputs = dict(
    cgpa=cgpa, python_skill=int(python), dsa_skill=int(dsa),
    ml_skill=int(ml), web_dev_skill=int(webdev),
    coding_score=float(coding), communication_score=float(comm),
    aptitude_score=float(apt), internships=internships,
    projects=projects, backlogs=backlogs,
    resume_score=float(resume), skill_score=skill_score,
    branch=branch, college_tier=str(tier),
)

if "result" not in st.session_state:
    st.session_state.result = predict(inputs)

if predict_btn:
    st.session_state.result = predict(inputs)

res = st.session_state.result

# ────────────────────────────────────────────────────────────────────────────
# Results panel
# ────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["  🎯  Prediction Results  ", "  📈  Analysis & Charts  "])

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

    # ── Profile summary ───────────────────────────────────────────────────────
    st.markdown("<div class='section-head'>Profile Summary</div>", unsafe_allow_html=True)

    skills_on = []
    if python: skills_on.append("Python")
    if dsa:    skills_on.append("DSA")
    if ml:     skills_on.append("Machine Learning")
    if webdev: skills_on.append("Web Development")
    skills_str = " · ".join(skills_on) if skills_on else "None selected"

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


# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-head'>Skill Radar vs Dataset Average</div>", unsafe_allow_html=True)

    # ── Radar chart ───────────────────────────────────────────────────────────
    col_l, col_r = st.columns([1, 1])

    with col_l:
        skill_labels = ["Python", "DSA", "ML", "Web Dev"]
        student_vals = [int(python), int(dsa), int(ml), int(webdev)]
        avg_vals     = [0.65, 0.56, 0.30, 0.39]   # dataset means from training run

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
        # ── Score breakdown bar chart ─────────────────────────────────────────
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
    # Background arc
    theta = np.linspace(np.pi, 0, 300)
    ax3.plot(np.cos(theta), np.sin(theta), color="#1f2330", linewidth=18, solid_capstyle="round")
    # Coloured fill
    fill_theta = np.linspace(np.pi, np.pi - prob*np.pi, 300)
    grad_color = "#22c55e" if prob > 0.7 else ("#f59e0b" if prob > 0.4 else "#ef4444")
    ax3.plot(np.cos(fill_theta), np.sin(fill_theta), color=grad_color, linewidth=18, solid_capstyle="round")
    # Needle
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
    for bar, val, lbl in zip(bars, vals, labels):
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

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#374151;font-size:0.78rem;margin-top:2rem;padding-top:1rem;border-top:1px solid #1f2330;'>
  Software Career Predictor · Trained on 9,000 students · 3 ML models · Built with Streamlit
</div>
""", unsafe_allow_html=True)
