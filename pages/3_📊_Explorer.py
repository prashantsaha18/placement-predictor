"""
pages/3_📊_Explorer.py — Dataset Explorer
"""
import sys, pathlib, warnings
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Dataset Explorer · SCP",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from utils.helpers import inject_css, load_dataset, section_head, metric_pill, ROLE_COLORS

inject_css()

DATASET = load_dataset()

st.markdown("""
<div style='background:linear-gradient(135deg,#0f1218,#0a0e1a);border:1px solid #1f2a40;
            border-radius:16px;padding:1.8rem 2.4rem;margin-bottom:1.5rem;'>
  <p style='font-family:Syne,sans-serif;font-size:0.75rem;font-weight:700;
             letter-spacing:0.18em;text-transform:uppercase;color:#7c83f5;margin:0 0 0.5rem;'>9,000 Students</p>
  <h1 style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#fff;margin:0;'>📊 Dataset Explorer</h1>
  <p style='color:#6b7280;font-size:0.9rem;margin:0.6rem 0 0;'>Explore the full training dataset. Use filters to drill into subgroups.</p>
</div>
""", unsafe_allow_html=True)

if DATASET is None:
    st.warning("Dataset CSV not found. Place `student_placement_salary_elite_v2.csv` in the project root.")
    st.stop()

df_full = DATASET.copy()

# ── Filters ───────────────────────────────────────────────────────────────────
with st.expander("🔍 Filter Dataset", expanded=False):
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_branches = st.multiselect("Branch", df_full["branch"].unique().tolist(),
                                       default=df_full["branch"].unique().tolist())
    with fc2:
        sel_tiers = st.multiselect("College Tier", ["1","2","3"], default=["1","2","3"])
    with fc3:
        cgpa_range = st.slider("CGPA Range", 5.0, 10.0, (5.0, 10.0), 0.1)

df = df_full[
    df_full["branch"].isin(sel_branches) &
    df_full["college_tier"].astype(str).isin(sel_tiers) &
    (df_full["cgpa"] >= cgpa_range[0]) &
    (df_full["cgpa"] <= cgpa_range[1])
].copy()

if df.empty:
    st.warning("No students match your filters. Please broaden the selection.")
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown(section_head("Dataset Overview"), unsafe_allow_html=True)
k1,k2,k3,k4,k5 = st.columns(5)
for col, num, lbl in zip(
    [k1,k2,k3,k4,k5],
    [f"{len(df):,}", f"{df['placed'].mean()*100:.1f}%",
     f"Rs{df['salary_lpa'].median():.1f}L", f"{df['cgpa'].mean():.2f}",
     f"{df['internships'].mean():.2f}"],
    ["Students","Placed","Median Salary","Avg CGPA","Avg Internships"]
):
    with col:
        st.markdown(metric_pill(num, lbl), unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Row 1: Placement by tier + branch ─────────────────────────────────────────
st.markdown(section_head("Placement Rates by Tier & Branch"), unsafe_allow_html=True)
dc1, dc2 = st.columns(2)
with dc1:
    tier_grp = df.groupby("college_tier")["placed"].mean()*100
    fig_t, ax_t = plt.subplots(figsize=(5,3.5))
    fig_t.patch.set_facecolor("#13161e"); ax_t.set_facecolor("#0d0f14")
    brs_t = ax_t.bar([f"Tier {t}" for t in tier_grp.index], tier_grp.values,
                      color=["#7c83f5","#5b63e8","#3a42d0"], edgecolor="#0d0f14", zorder=3)
    for b,v in zip(brs_t, tier_grp.values):
        ax_t.text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.1f}%",
                  ha="center", color="white", fontsize=9, fontweight="bold")
    ax_t.set_ylim(0,105); ax_t.set_ylabel("Placement %", color="#6b7280", fontsize=9)
    ax_t.tick_params(colors="#9ca3af"); ax_t.spines[:].set_color("#1f2330")
    ax_t.grid(axis="y", color="#1f2330", linewidth=0.5, zorder=0)
    ax_t.set_title("By College Tier", color="#9ca3af", fontsize=10, pad=8)
    plt.tight_layout(); st.pyplot(fig_t, use_container_width=True); plt.close(fig_t)

with dc2:
    br_grp = df.groupby("branch")["placed"].mean().sort_values(ascending=False)*100
    fig_b, ax_b = plt.subplots(figsize=(5,3.5))
    fig_b.patch.set_facecolor("#13161e"); ax_b.set_facecolor("#0d0f14")
    col_b = ["#7c83f5"] + ["#2a2f45"]*(len(br_grp)-1)
    brs_b = ax_b.bar(br_grp.index, br_grp.values, color=col_b, edgecolor="#0d0f14", zorder=3)
    for b,v in zip(brs_b, br_grp.values):
        ax_b.text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.0f}%",
                  ha="center", color="white", fontsize=8.5, fontweight="bold")
    ax_b.set_ylim(0,105); ax_b.set_ylabel("Placement %", color="#6b7280", fontsize=9)
    ax_b.tick_params(colors="#9ca3af", labelsize=8.5); ax_b.spines[:].set_color("#1f2330")
    ax_b.grid(axis="y", color="#1f2330", linewidth=0.5, zorder=0)
    ax_b.set_title("By Branch", color="#9ca3af", fontsize=10, pad=8)
    plt.tight_layout(); st.pyplot(fig_b, use_container_width=True); plt.close(fig_b)

# ── Row 2: Salary dist + Role pie ─────────────────────────────────────────────
st.markdown(section_head("Salary & Job Role Distribution"), unsafe_allow_html=True)
dc3, dc4 = st.columns(2)
with dc3:
    fig_s, ax_s = plt.subplots(figsize=(5,3.5))
    fig_s.patch.set_facecolor("#13161e"); ax_s.set_facecolor("#0d0f14")
    ax_s.hist(df["salary_lpa"], bins=30, color="#7c83f5", edgecolor="#0d0f14", alpha=0.85, zorder=3)
    ax_s.axvline(df["salary_lpa"].median(), color="#22c55e", linewidth=2,
                  linestyle="--", label=f"Median Rs{df['salary_lpa'].median():.1f}L", zorder=4)
    ax_s.axvline(df["salary_lpa"].mean(), color="#f59e0b", linewidth=1.5,
                  linestyle=":", label=f"Mean Rs{df['salary_lpa'].mean():.1f}L", zorder=4)
    ax_s.set_xlabel("Salary (LPA)", color="#6b7280", fontsize=9)
    ax_s.set_ylabel("Students", color="#6b7280", fontsize=9)
    ax_s.tick_params(colors="#9ca3af"); ax_s.spines[:].set_color("#1f2330")
    ax_s.grid(axis="y", color="#1f2330", linewidth=0.5, zorder=0)
    ax_s.legend(fontsize=8.5, labelcolor="#9ca3af", facecolor="#13161e", edgecolor="#1f2330")
    ax_s.set_title("Salary Distribution", color="#9ca3af", fontsize=10, pad=8)
    plt.tight_layout(); st.pyplot(fig_s, use_container_width=True); plt.close(fig_s)

with dc4:
    rc = df["job_role"].value_counts()
    fig_p, ax_p = plt.subplots(figsize=(5,3.5))
    fig_p.patch.set_facecolor("#13161e"); ax_p.set_facecolor("#0d0f14")
    role_pal = ["#7c83f5","#22c55e","#f59e0b","#ef4444"]
    wedges, texts, autotexts = ax_p.pie(
        rc.values, labels=rc.index, colors=role_pal, autopct="%1.0f%%",
        startangle=140, textprops={"color":"#9ca3af","fontsize":9},
        wedgeprops={"edgecolor":"#0d0f14","linewidth":1.5}
    )
    for at in autotexts: at.set_color("white"); at.set_fontweight("bold")
    ax_p.set_title("Job Role Distribution", color="#9ca3af", fontsize=10, pad=8)
    plt.tight_layout(); st.pyplot(fig_p, use_container_width=True); plt.close(fig_p)

# ── Row 3: CGPA band + Salary by role ─────────────────────────────────────────
st.markdown(section_head("CGPA vs Placement & Salary by Role"), unsafe_allow_html=True)
dc5, dc6 = st.columns(2)
with dc5:
    cgpa_bins = pd.cut(df["cgpa"], bins=[5,6,7,8,9,10], labels=["5-6","6-7","7-8","8-9","9-10"])
    cgpa_pl = df.groupby(cgpa_bins, observed=True)["placed"].mean()*100
    fig_c, ax_c = plt.subplots(figsize=(5,3.5))
    fig_c.patch.set_facecolor("#13161e"); ax_c.set_facecolor("#0d0f14")
    ax_c.plot(cgpa_pl.index, cgpa_pl.values, color="#7c83f5",
               marker="o", linewidth=2.5, markersize=7, zorder=3)
    ax_c.fill_between(range(len(cgpa_pl)), cgpa_pl.values, color="#7c83f5", alpha=0.15)
    ax_c.set_xticks(range(len(cgpa_pl)))
    ax_c.set_xticklabels(cgpa_pl.index, color="#9ca3af", fontsize=9)
    ax_c.set_ylabel("Placement %", color="#6b7280", fontsize=9)
    ax_c.tick_params(colors="#9ca3af"); ax_c.spines[:].set_color("#1f2330")
    ax_c.grid(color="#1f2330", linewidth=0.5, zorder=0)
    ax_c.set_title("Placement Rate by CGPA Band", color="#9ca3af", fontsize=10, pad=8)
    plt.tight_layout(); st.pyplot(fig_c, use_container_width=True); plt.close(fig_c)

with dc6:
    sal_by_role = df.groupby("job_role")["salary_lpa"].median()
    fig_sr, ax_sr = plt.subplots(figsize=(5,3.5))
    fig_sr.patch.set_facecolor("#13161e"); ax_sr.set_facecolor("#0d0f14")
    colors_sr = [ROLE_COLORS.get(r,"#7c83f5") for r in sal_by_role.index]
    brs_sr = ax_sr.bar(sal_by_role.index, sal_by_role.values,
                        color=colors_sr, edgecolor="#0d0f14", zorder=3)
    for b,v in zip(brs_sr, sal_by_role.values):
        ax_sr.text(b.get_x()+b.get_width()/2, v+0.3, f"Rs{v:.0f}L",
                   ha="center", color="white", fontsize=8.5, fontweight="bold")
    ax_sr.set_ylabel("Median Salary (LPA)", color="#6b7280", fontsize=9)
    ax_sr.tick_params(colors="#9ca3af", labelsize=8.5); ax_sr.spines[:].set_color("#1f2330")
    ax_sr.grid(axis="y", color="#1f2330", linewidth=0.5, zorder=0)
    ax_sr.set_title("Median Salary by Job Role", color="#9ca3af", fontsize=10, pad=8)
    plt.tight_layout(); st.pyplot(fig_sr, use_container_width=True); plt.close(fig_sr)

# ── Internships vs placement ───────────────────────────────────────────────────
st.markdown(section_head("Internship Impact & Skill Correlation"), unsafe_allow_html=True)
dc7, dc8 = st.columns(2)
with dc7:
    int_grp = df.groupby("internships")["placed"].mean()*100
    fig_i, ax_i = plt.subplots(figsize=(5,3.5))
    fig_i.patch.set_facecolor("#13161e"); ax_i.set_facecolor("#0d0f14")
    brs_i = ax_i.bar(int_grp.index, int_grp.values,
                      color=["#ef4444","#f59e0b","#7c83f5","#22c55e"],
                      edgecolor="#0d0f14", zorder=3)
    for b,v in zip(brs_i, int_grp.values):
        ax_i.text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.1f}%",
                  ha="center", color="white", fontsize=9, fontweight="bold")
    ax_i.set_xlabel("Number of Internships", color="#6b7280", fontsize=9)
    ax_i.set_ylabel("Placement %", color="#6b7280", fontsize=9)
    ax_i.set_ylim(0,105); ax_i.tick_params(colors="#9ca3af")
    ax_i.spines[:].set_color("#1f2330")
    ax_i.grid(axis="y", color="#1f2330", linewidth=0.5, zorder=0)
    ax_i.set_title("Placement Rate by Internship Count", color="#9ca3af", fontsize=10, pad=8)
    plt.tight_layout(); st.pyplot(fig_i, use_container_width=True); plt.close(fig_i)

with dc8:
    skill_rates = {
        "Python": df["python_skill"].mean()*100,
        "DSA":    df["dsa_skill"].mean()*100,
        "ML":     df["ml_skill"].mean()*100,
        "Web Dev":df["web_dev_skill"].mean()*100,
    }
    placed_skill = {
        "Python": df[df["placed"]==1]["python_skill"].mean()*100,
        "DSA":    df[df["placed"]==1]["dsa_skill"].mean()*100,
        "ML":     df[df["placed"]==1]["ml_skill"].mean()*100,
        "Web Dev":df[df["placed"]==1]["web_dev_skill"].mean()*100,
    }
    fig_sk, ax_sk = plt.subplots(figsize=(5,3.5))
    fig_sk.patch.set_facecolor("#13161e"); ax_sk.set_facecolor("#0d0f14")
    xsk = np.arange(4)
    ax_sk.bar(xsk-0.2, list(skill_rates.values()), 0.35,
               color="#2a2f45", label="All Students", zorder=3)
    ax_sk.bar(xsk+0.2, list(placed_skill.values()), 0.35,
               color="#22c55e", label="Placed Only", zorder=3)
    ax_sk.set_xticks(xsk)
    ax_sk.set_xticklabels(list(skill_rates.keys()), color="#9ca3af", fontsize=9)
    ax_sk.set_ylabel("% with Skill", color="#6b7280", fontsize=9)
    ax_sk.set_ylim(0,115); ax_sk.tick_params(colors="#9ca3af")
    ax_sk.spines[:].set_color("#1f2330")
    ax_sk.grid(axis="y", color="#1f2330", linewidth=0.5, zorder=0)
    ax_sk.legend(fontsize=9, labelcolor="#9ca3af", facecolor="#13161e", edgecolor="#1f2330")
    ax_sk.set_title("Skill Adoption: All vs Placed", color="#9ca3af", fontsize=10, pad=8)
    plt.tight_layout(); st.pyplot(fig_sk, use_container_width=True); plt.close(fig_sk)

# ── Raw data table ─────────────────────────────────────────────────────────────
with st.expander("📋 View Raw Data Sample (first 100 rows)"):
    st.dataframe(
        df.head(100).style.background_gradient(
            subset=["cgpa","salary_lpa","coding_score"],
            cmap="Blues"
        ),
        use_container_width=True
    )
