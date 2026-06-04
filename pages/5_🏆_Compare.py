"""
pages/5_🏆_Compare.py — Peer Comparison Engine
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
    page_title="Peer Comparison · SCP",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.helpers import (
    inject_css, load_dataset, load_models, predict, compute_readiness,
    sidebar_profile_form, section_head, metric_pill, ROLE_COLORS,
)

inject_css()

DATASET = load_dataset()
try:
    models = load_models()
    MODELS_READY = True
except Exception as e:
    MODELS_READY = False

inputs, predict_btn = sidebar_profile_form()

st.markdown("""
<div style='background:linear-gradient(135deg,#0f1218,#0a0e1a);border:1px solid #1f2a40;
            border-radius:16px;padding:1.8rem 2.4rem;margin-bottom:1.5rem;'>
  <p style='font-family:Syne,sans-serif;font-size:0.75rem;font-weight:700;
             letter-spacing:0.18em;text-transform:uppercase;color:#f59e0b;margin:0 0 0.5rem;'>Benchmarking</p>
  <h1 style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#fff;margin:0;'>🏆 Peer Comparison Engine</h1>
  <p style='color:#6b7280;font-size:0.9rem;margin:0.6rem 0 0;'>
    Compare your profile against students who share your branch and college tier.
    See where you really stand.
  </p>
</div>
""", unsafe_allow_html=True)

if DATASET is None:
    st.warning("Dataset not found.")
    st.stop()

if not MODELS_READY:
    st.error("Models not loaded.")
    st.stop()

import time
if "result" not in st.session_state or predict_btn:
    with st.spinner("Analyzing your peer group…"):
        if predict_btn: time.sleep(0.4)
        st.session_state.result = predict(inputs, models)

res = st.session_state.result
readiness = compute_readiness(inputs, res["prob"])

df = DATASET.copy()

# ── Build peer group ──────────────────────────────────────────────────────────
df["college_tier"] = df["college_tier"].astype(str)
peer_df = df[
    (df["branch"] == inputs["branch"]) &
    (df["college_tier"] == inputs["college_tier"])
].copy()

if len(peer_df) < 10:
    peer_df = df[df["branch"] == inputs["branch"]].copy()
    peer_note = f"(Expanded to all {inputs['branch']} students due to small sample)"
else:
    peer_note = f"({inputs['branch']} · Tier {inputs['college_tier']})"

# ── Overview cards ────────────────────────────────────────────────────────────
st.markdown(section_head(f"Your Peer Group {peer_note}"), unsafe_allow_html=True)

n_peers = len(peer_df)
peer_place_rate = peer_df["placed"].mean() * 100
peer_median_sal = peer_df["salary_lpa"].median()
peer_avg_cgpa   = peer_df["cgpa"].mean()
peer_avg_int    = peer_df["internships"].mean()

ov1, ov2, ov3, ov4, ov5 = st.columns(5)
for col, num, lbl in zip(
    [ov1, ov2, ov3, ov4, ov5],
    [f"{n_peers:,}", f"{peer_place_rate:.1f}%",
     f"₹{peer_median_sal:.1f}L", f"{peer_avg_cgpa:.2f}",
     f"{peer_avg_int:.1f}"],
    ["Peers in Group", "Peer Placement Rate", "Peer Median Salary",
     "Peer Avg CGPA", "Peer Avg Internships"]
):
    with col:
        st.markdown(metric_pill(num, lbl), unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── You vs Peers ──────────────────────────────────────────────────────────────
st.markdown(section_head("📊 You vs Your Peers"), unsafe_allow_html=True)

# Percentile computations
def pct_rank(series, val):
    return (series < val).mean() * 100

cgpa_pct  = pct_rank(peer_df["cgpa"], inputs["cgpa"])
code_pct  = pct_rank(peer_df["coding_score"], inputs["coding_score"])
apt_pct   = pct_rank(peer_df["aptitude_score"], inputs["aptitude_score"])
res_pct   = pct_rank(peer_df["resume_score"], inputs["resume_score"])
prob_pct  = pct_rank(peer_df["placed"], res["prob"])

comparisons = [
    ("CGPA",          inputs["cgpa"],                   peer_avg_cgpa,          cgpa_pct,  "/10.0"),
    ("Coding Score",  inputs["coding_score"],            peer_df["coding_score"].mean(), code_pct, "/100"),
    ("Aptitude",      inputs["aptitude_score"],          peer_df["aptitude_score"].mean(), apt_pct, "/100"),
    ("Resume Score",  inputs["resume_score"],            peer_df["resume_score"].mean(), res_pct, "/100"),
    ("Internships",   inputs["internships"],             peer_avg_int,           pct_rank(peer_df["internships"], inputs["internships"]), ""),
    ("Projects",      inputs["projects"],                peer_df["projects"].mean(), pct_rank(peer_df["projects"], inputs["projects"]), ""),
]

vc1, vc2 = st.columns(2)
for i, (label, your_val, peer_avg, pct, unit) in enumerate(comparisons):
    col = vc1 if i % 2 == 0 else vc2
    color = "#22c55e" if pct >= 60 else ("#f59e0b" if pct >= 40 else "#ef4444")
    with col:
        st.markdown(f"""
        <div style='background:#13161e;border:1px solid #1f2330;border-radius:12px;
                    padding:1rem 1.4rem;margin-bottom:0.8rem;'>
          <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.6rem;'>
            <div>
              <div style='font-family:Syne,sans-serif;font-size:0.8rem;font-weight:700;
                           color:#c7caff;margin-bottom:0.1rem;'>{label}</div>
              <div style='font-size:0.75rem;color:#4b5563;'>Peer avg: {peer_avg:.1f}{unit}</div>
            </div>
            <div style='text-align:right;'>
              <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:800;color:{color};'>{your_val}{unit}</div>
              <div style='font-size:0.72rem;color:{color};'>Top {100-pct:.0f}%</div>
            </div>
          </div>
          <div class='bar-bg'>
            <div class='bar-fill' style='width:{pct:.0f}%;background:{color};'></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Peer charts ───────────────────────────────────────────────────────────────
st.markdown(section_head("📈 Peer Distribution Charts"), unsafe_allow_html=True)
cc1, cc2 = st.columns(2)

with cc1:
    # CGPA distribution with your marker
    fig_cg, ax_cg = plt.subplots(figsize=(5, 3.5))
    fig_cg.patch.set_facecolor("#13161e"); ax_cg.set_facecolor("#0d0f14")
    ax_cg.hist(peer_df["cgpa"], bins=20, color="#2a2f45", edgecolor="#0d0f14", alpha=0.85, zorder=3)
    ax_cg.axvline(inputs["cgpa"], color="#7c83f5", linewidth=2.5,
                   linestyle="--", label=f"You: {inputs['cgpa']:.1f}", zorder=4)
    ax_cg.axvline(peer_avg_cgpa, color="#22c55e", linewidth=1.5,
                   linestyle=":", label=f"Peer Avg: {peer_avg_cgpa:.2f}", zorder=4)
    ax_cg.set_xlabel("CGPA", color="#6b7280", fontsize=9)
    ax_cg.set_ylabel("Count", color="#6b7280", fontsize=9)
    ax_cg.tick_params(colors="#9ca3af"); ax_cg.spines[:].set_color("#1f2330")
    ax_cg.grid(axis="y", color="#1f2330", linewidth=0.5, zorder=0)
    ax_cg.legend(fontsize=8.5, labelcolor="#9ca3af", facecolor="#13161e", edgecolor="#1f2330")
    ax_cg.set_title(f"CGPA Distribution — Your Peer Group", color="#9ca3af", fontsize=10, pad=8)
    plt.tight_layout(); st.pyplot(fig_cg, use_container_width=True); plt.close(fig_cg)

with cc2:
    # Salary distribution with your marker
    fig_sal, ax_sal = plt.subplots(figsize=(5, 3.5))
    fig_sal.patch.set_facecolor("#13161e"); ax_sal.set_facecolor("#0d0f14")
    placed_sal = peer_df[peer_df["placed"] == 1]["salary_lpa"]
    ax_sal.hist(placed_sal, bins=20, color="#2a2f45", edgecolor="#0d0f14", alpha=0.85, zorder=3)
    ax_sal.axvline(res["salary"], color="#f59e0b", linewidth=2.5,
                    linestyle="--", label=f"Your Est: ₹{res['salary']:.1f}L", zorder=4)
    ax_sal.axvline(placed_sal.median(), color="#22c55e", linewidth=1.5,
                    linestyle=":", label=f"Peer Median: ₹{placed_sal.median():.1f}L", zorder=4)
    ax_sal.set_xlabel("Salary (LPA)", color="#6b7280", fontsize=9)
    ax_sal.set_ylabel("Placed Students", color="#6b7280", fontsize=9)
    ax_sal.tick_params(colors="#9ca3af"); ax_sal.spines[:].set_color("#1f2330")
    ax_sal.grid(axis="y", color="#1f2330", linewidth=0.5, zorder=0)
    ax_sal.legend(fontsize=8.5, labelcolor="#9ca3af", facecolor="#13161e", edgecolor="#1f2330")
    ax_sal.set_title("Salary Distribution — Placed Peers", color="#9ca3af", fontsize=10, pad=8)
    plt.tight_layout(); st.pyplot(fig_sal, use_container_width=True); plt.close(fig_sal)

# ── Top performers in peer group ──────────────────────────────────────────────
st.markdown(section_head("⭐ Top Performers in Your Peer Group"), unsafe_allow_html=True)
top_peers = (peer_df[peer_df["placed"] == 1]
             .nlargest(5, "salary_lpa")[["cgpa","coding_score","aptitude_score",
                                          "internships","projects","salary_lpa","job_role"]]
             .reset_index(drop=True))
top_peers.index += 1
top_peers.columns = ["CGPA","Coding","Aptitude","Internships","Projects","Salary (LPA)","Job Role"]
st.dataframe(
    top_peers.style.background_gradient(subset=["CGPA","Salary (LPA)"], cmap="Blues"),
    use_container_width=True
)

# ── Readiness rank ────────────────────────────────────────────────────────────
st.markdown(section_head("🏅 Your Overall Readiness Rank"), unsafe_allow_html=True)
r1, r2 = st.columns([1, 2])
with r1:
    tier_bg = {
        "Platinum": "linear-gradient(135deg,#334155,#64748b)",
        "Gold":     "linear-gradient(135deg,#78350f,#d97706)",
        "Silver":   "linear-gradient(135deg,#334155,#64748b)",
        "Bronze":   "linear-gradient(135deg,#451a03,#92400e)",
    }.get(readiness["tier"])
    st.markdown(f"""
    <div class="badge-wrap">
      <div class="badge-circle {readiness['glow']}" style="background:{tier_bg};">
        <div class="badge-score">{readiness['score']:.0f}</div>
        <div class="badge-lbl">/ 100</div>
      </div>
      <div class="badge-tier" style="color:{readiness['color']};">{readiness['tier']} Tier</div>
      <div class="badge-pct">Top {100-readiness['pct']:.0f}% of dataset</div>
    </div>""", unsafe_allow_html=True)

with r2:
    st.markdown(f"""
    <div style='background:#13161e;border:1px solid #1f2330;border-radius:12px;
                padding:1.4rem 1.8rem;height:100%;'>
      <div style='font-family:Syne,sans-serif;font-size:0.88rem;font-weight:700;
                   color:#c7caff;margin-bottom:0.8rem;'>Peer Group Summary</div>
      <table style='width:100%;font-size:0.83rem;border-collapse:collapse;'>
        <tr style='color:#4b5563;'>
          <td style='padding:0.3rem 0;'>Your Placement Probability</td>
          <td style='text-align:right;font-weight:700;color:#7c83f5;'>{res['prob']*100:.1f}%</td>
        </tr>
        <tr style='color:#4b5563;'>
          <td style='padding:0.3rem 0;'>Peer Placement Rate</td>
          <td style='text-align:right;font-weight:700;color:#9ca3af;'>{peer_place_rate:.1f}%</td>
        </tr>
        <tr style='color:#4b5563;'>
          <td style='padding:0.3rem 0;'>Your CGPA Percentile</td>
          <td style='text-align:right;font-weight:700;color:#7c83f5;'>Top {100-cgpa_pct:.0f}%</td>
        </tr>
        <tr style='color:#4b5563;'>
          <td style='padding:0.3rem 0;'>Your Coding Percentile</td>
          <td style='text-align:right;font-weight:700;color:#7c83f5;'>Top {100-code_pct:.0f}%</td>
        </tr>
        <tr style='color:#4b5563;'>
          <td style='padding:0.3rem 0;'>Peer Group Size</td>
          <td style='text-align:right;font-weight:700;color:#9ca3af;'>{n_peers:,} students</td>
        </tr>
      </table>
    </div>
    """, unsafe_allow_html=True)
