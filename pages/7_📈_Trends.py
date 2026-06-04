"""
pages/7_📈_Trends.py — Industry Trends Dashboard
"""
import sys, pathlib, warnings
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Industry Trends · SCP",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from utils.helpers import inject_css, section_head, metric_pill, ROLE_COLORS

inject_css()

st.markdown("""
<div style='background:linear-gradient(135deg,#0f1218,#0a0e1a);border:1px solid #1f2a40;
            border-radius:16px;padding:1.8rem 2.4rem;margin-bottom:1.5rem;'>
  <p style='font-family:Syne,sans-serif;font-size:0.75rem;font-weight:700;
             letter-spacing:0.18em;text-transform:uppercase;color:#38bdf8;margin:0 0 0.5rem;'>Industry Intelligence</p>
  <h1 style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#fff;margin:0;'>📈 Industry Trends</h1>
  <p style='color:#6b7280;font-size:0.9rem;margin:0.6rem 0 0;'>
    Explore salary trends, in-demand skills, hiring patterns, and the tech job market landscape.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Live KPIs ─────────────────────────────────────────────────────────────────
st.markdown(section_head("🌍 2024 India Tech Job Market Snapshot"), unsafe_allow_html=True)
k1,k2,k3,k4,k5 = st.columns(5)
for col, num, lbl in zip(
    [k1,k2,k3,k4,k5],
    ["₹18–45L", "34%", "₹12L", "72%", "1.4M+"],
    ["Fresher SDE Avg Pkg", "YoY Demand Growth", "Non-CS Branch Avg", "Open-to-Remote", "Tech Jobs Posted"]
):
    with col:
        st.markdown(metric_pill(num, lbl), unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Salary Trends by Role (2019–2024) ─────────────────────────────────────────
st.markdown(section_head("💰 Salary Trends by Role (2019–2024)"), unsafe_allow_html=True)

years = [2019, 2020, 2021, 2022, 2023, 2024]
salary_trends = {
    "Data Scientist":    [8.5,  8.2,  11.5, 15.0, 16.8, 18.2],
    "Software Engineer": [7.2,  7.0,  9.5,  13.5, 14.8, 16.5],
    "Web Developer":     [5.5,  5.2,  7.0,  9.5,  10.5, 11.8],
    "Analyst":           [5.0,  4.8,  6.5,  8.0,  9.2,  10.5],
}

fig_tr, ax_tr = plt.subplots(figsize=(10, 4.5))
fig_tr.patch.set_facecolor("#13161e"); ax_tr.set_facecolor("#0d0f14")

for role, vals in salary_trends.items():
    color = ROLE_COLORS.get(role, "#7c83f5")
    ax_tr.plot(years, vals, marker="o", linewidth=2.5, markersize=7,
                color=color, label=role, zorder=3)
    ax_tr.fill_between(years, vals, alpha=0.08, color=color)
    ax_tr.annotate(f"₹{vals[-1]}L", (years[-1], vals[-1]),
                    xytext=(5, 2), textcoords="offset points",
                    color=color, fontsize=8.5, fontweight="bold")

ax_tr.set_xlabel("Year", color="#6b7280", fontsize=9)
ax_tr.set_ylabel("Median Salary (LPA)", color="#6b7280", fontsize=9)
ax_tr.set_xticks(years); ax_tr.set_xticklabels(years, color="#9ca3af")
ax_tr.tick_params(colors="#9ca3af"); ax_tr.spines[:].set_color("#1f2330")
ax_tr.grid(color="#1f2330", linewidth=0.5, alpha=0.7, zorder=0)
ax_tr.legend(fontsize=9, labelcolor="#9ca3af", facecolor="#13161e",
              edgecolor="#1f2330", loc="upper left")
ax_tr.set_title("Entry-Level Salary Trends — India Tech Industry",
                color="#9ca3af", fontsize=11, pad=10)
plt.tight_layout(); st.pyplot(fig_tr, use_container_width=True); plt.close(fig_tr)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Row 2: In-demand skills + Remote/Onsite ────────────────────────────────────
st.markdown(section_head("🔥 In-Demand Skills & Work Modes"), unsafe_allow_html=True)
tr1, tr2 = st.columns(2)

with tr1:
    skills_demand = {
        "Python":            94,
        "SQL":               88,
        "React / Next.js":   82,
        "DSA (LeetCode)":    91,
        "Machine Learning":  76,
        "System Design":     85,
        "TypeScript":        70,
        "Docker / K8s":      68,
        "Cloud (AWS/GCP)":   73,
        "GenAI / LLMs":      65,
    }
    sorted_skills = sorted(skills_demand.items(), key=lambda x: -x[1])
    skill_names = [s[0] for s in sorted_skills]
    skill_vals  = [s[1] for s in sorted_skills]

    fig_sk, ax_sk = plt.subplots(figsize=(5.5, 5))
    fig_sk.patch.set_facecolor("#13161e"); ax_sk.set_facecolor("#0d0f14")
    norm_vals = [(v - 60) / 40 for v in skill_vals]
    colors_sk = plt.cm.YlOrRd(np.array(norm_vals) * 0.7 + 0.3)
    brs_sk = ax_sk.barh(skill_names, skill_vals, color=colors_sk,
                         edgecolor="#0d0f14", height=0.65, zorder=3)
    for b, v in zip(brs_sk, skill_vals):
        ax_sk.text(v + 0.5, b.get_y() + b.get_height()/2,
                   f"{v}%", va="center", color="white",
                   fontsize=8.5, fontweight="bold")
    ax_sk.set_xlim(50, 100)
    ax_sk.set_xlabel("% of Job Postings Mentioning Skill", color="#6b7280", fontsize=9)
    ax_sk.tick_params(colors="#9ca3af", labelsize=9); ax_sk.spines[:].set_color("#1f2330")
    ax_sk.grid(axis="x", color="#1f2330", linewidth=0.5, zorder=0)
    ax_sk.set_title("Most In-Demand Skills (2024)", color="#9ca3af", fontsize=10, pad=8)
    ax_sk.invert_yaxis()
    plt.tight_layout(); st.pyplot(fig_sk, use_container_width=True); plt.close(fig_sk)

with tr2:
    # Work mode trends
    modes_years = [2019, 2020, 2021, 2022, 2023, 2024]
    onsite     = [95, 30, 15, 40, 55, 60]
    remote     = [3,  65, 80, 45, 28, 20]
    hybrid     = [2,  5,  5,  15, 17, 20]

    fig_wm, ax_wm = plt.subplots(figsize=(5.5, 5))
    fig_wm.patch.set_facecolor("#13161e"); ax_wm.set_facecolor("#0d0f14")

    ax_wm.stackplot(modes_years,
                     [onsite, hybrid, remote],
                     labels=["On-site", "Hybrid", "Remote"],
                     colors=["#7c83f5", "#f59e0b", "#22c55e"],
                     alpha=0.85)
    ax_wm.set_xlabel("Year", color="#6b7280", fontsize=9)
    ax_wm.set_ylabel("% of Jobs", color="#6b7280", fontsize=9)
    ax_wm.set_xticks(modes_years); ax_wm.set_xticklabels(modes_years, color="#9ca3af")
    ax_wm.tick_params(colors="#9ca3af"); ax_wm.spines[:].set_color("#1f2330")
    ax_wm.set_ylim(0, 100)
    ax_wm.legend(loc="upper right", fontsize=9, labelcolor="#9ca3af",
                  facecolor="#13161e", edgecolor="#1f2330")
    ax_wm.set_title("Remote / Hybrid / On-site Trends", color="#9ca3af", fontsize=10, pad=8)
    ax_wm.grid(color="#1f2330", linewidth=0.4, alpha=0.6)
    plt.tight_layout(); st.pyplot(fig_wm, use_container_width=True); plt.close(fig_wm)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Top Hiring Companies ───────────────────────────────────────────────────────
st.markdown(section_head("🏢 Top Hiring Companies by Role"), unsafe_allow_html=True)

COMPANIES = {
    "Data Scientist": [
        ("Google DeepMind", "₹35–80L", "Research + Applied"),
        ("Amazon",          "₹25–55L", "Applied ML / Rec systems"),
        ("Flipkart",        "₹20–45L", "E-commerce ML"),
        ("ShareChat",       "₹18–40L", "Social media AI"),
        ("Meesho",          "₹15–35L", "Growth / Supply chain ML"),
    ],
    "Software Engineer": [
        ("Microsoft",       "₹30–75L", "Azure / Cloud / OS"),
        ("Goldman Sachs",   "₹25–60L", "Quant / FinTech"),
        ("Atlassian",       "₹30–65L", "Developer tools"),
        ("Razorpay",        "₹20–50L", "FinTech / Payments"),
        ("Zepto",           "₹18–45L", "High-scale logistics"),
    ],
    "Web Developer": [
        ("Hotstar / Star",  "₹15–35L", "Streaming platform"),
        ("Urban Company",   "₹14–30L", "Home services"),
        ("Myntra",          "₹14–32L", "Fashion e-commerce"),
        ("BookMyShow",      "₹12–28L", "Entertainment"),
        ("Nykaa",           "₹12–26L", "Beauty e-commerce"),
    ],
    "Analyst": [
        ("McKinsey",        "₹18–35L", "Strategy consulting"),
        ("Bain & Co.",      "₹18–35L", "Business analytics"),
        ("Groww",           "₹15–32L", "FinTech / Investments"),
        ("Paytm",           "₹12–28L", "Payments analytics"),
        ("Zomato",          "₹12–26L", "Food delivery ops"),
    ],
}

tc1, tc2 = st.columns(2)
for i, (role_, companies_) in enumerate(COMPANIES.items()):
    col = tc1 if i % 2 == 0 else tc2
    rc = ROLE_COLORS.get(role_, "#7c83f5")
    with col:
        st.markdown(f"""
        <div style='background:#13161e;border:1px solid #1f2330;border-left:3px solid {rc};
                    border-radius:12px;padding:1.1rem 1.4rem;margin-bottom:0.8rem;'>
          <div style='font-family:Syne,sans-serif;font-size:0.88rem;font-weight:700;
                       color:#c7caff;margin-bottom:0.7rem;'>{role_}</div>
        """, unsafe_allow_html=True)
        for company, pkg, domain in companies_:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;padding:0.35rem 0;
                        border-bottom:1px solid #1f2330;'>
              <div>
                <div style='font-size:0.82rem;color:#c7caff;font-weight:600;'>{company}</div>
                <div style='font-size:0.72rem;color:#4b5563;'>{domain}</div>
              </div>
              <div style='font-size:0.8rem;color:#22c55e;font-weight:700;white-space:nowrap;margin-left:0.5rem;'>{pkg}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── GenAI & Emerging Roles ────────────────────────────────────────────────────
st.markdown(section_head("🤖 Emerging Roles & GenAI Impact"), unsafe_allow_html=True)

emerging = [
    ("🤖", "AI/ML Engineer", "₹25–60L", "+340% demand in 2024",
     "Model fine-tuning, MLOps, vector databases, RAG systems"),
    ("📊", "Data Platform Engineer", "₹20–50L", "+180% demand",
     "dbt, Airflow, Spark, Kafka, data lake architecture"),
    ("🔐", "Security Engineer", "₹18–45L", "+120% demand",
     "Cloud security, DevSecOps, penetration testing"),
    ("☁️", "Cloud/DevOps Engineer", "₹18–40L", "+95% demand",
     "Kubernetes, Terraform, CI/CD, AWS/GCP/Azure"),
    ("🎨", "Product Designer (UX)", "₹14–35L", "+85% demand",
     "Figma, user research, design systems, prototyping"),
]

for icon, role_, pkg, growth, skills in emerging:
    st.markdown(f"""
    <div class='trend-card'>
      <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.5rem;'>
        <div style='display:flex;gap:0.7rem;align-items:center;'>
          <span style='font-size:1.6rem;'>{icon}</span>
          <div>
            <div style='font-family:Syne,sans-serif;font-size:0.92rem;font-weight:700;color:#c7caff;'>{role_}</div>
            <div style='font-size:0.78rem;color:#4b5563;'>{skills}</div>
          </div>
        </div>
        <div style='text-align:right;'>
          <div style='font-size:0.88rem;color:#22c55e;font-weight:700;'>{pkg}</div>
          <div style='font-size:0.72rem;color:#4b5563;'>{growth}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='font-size:0.72rem;color:#374151;text-align:center;margin-top:1.5rem;'>
  📊 Data is representative and based on India tech market trends (2019–2024).
  Salary ranges reflect fresher to 2-year experience bracket.
</div>
""", unsafe_allow_html=True)
