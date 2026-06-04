"""
pages/2_🔄_What_If.py — What-If Scenario Simulator
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
    page_title="What-If Simulator · SCP",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.helpers import (
    inject_css, load_models, predict, compute_whatif,
    sidebar_profile_form, section_head,
)

inject_css()

try:
    models = load_models()
    MODELS_READY = True
except Exception as e:
    MODELS_READY = False
    MODEL_ERROR = str(e)

inputs, predict_btn = sidebar_profile_form()

st.markdown("""
<div style='background:linear-gradient(135deg,#0f1218,#0a0e1a);border:1px solid #1f2a40;
            border-radius:16px;padding:1.8rem 2.4rem;margin-bottom:1.5rem;'>
  <p style='font-family:Syne,sans-serif;font-size:0.75rem;font-weight:700;
             letter-spacing:0.18em;text-transform:uppercase;color:#7c83f5;margin:0 0 0.5rem;'>Interactive Analysis</p>
  <h1 style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#fff;margin:0;'>🔄 What-If Simulator</h1>
  <p style='color:#6b7280;font-size:0.9rem;margin:0.6rem 0 0;'>
    See exactly how improving one aspect of your profile affects placement probability.
  </p>
</div>
""", unsafe_allow_html=True)

if not MODELS_READY:
    st.error(f"Models not loaded: {MODEL_ERROR}")
    st.stop()

import time
if "result" not in st.session_state or predict_btn:
    with st.spinner("Running predictions…"):
        if predict_btn: time.sleep(0.4)
        st.session_state.result = predict(inputs, models)

res = st.session_state.result
base_prob = res["prob"]

# ── Summary card ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:#13161e;border:1px solid #1f2330;border-radius:12px;
            padding:1.2rem 1.5rem;margin-bottom:1.4rem;'>
  <div style='font-family:Syne,sans-serif;font-size:0.9rem;font-weight:700;
               color:#c7caff;margin-bottom:0.3rem;'>Your Current Baseline</div>
  <div style='font-size:0.85rem;color:#6b7280;line-height:1.6;'>
    Placement probability: <b style='color:#7c83f5;font-size:1.2rem;'>{base_prob*100:.1f}%</b>
    &nbsp;·&nbsp; Each bar shows the delta from this value.
  </div>
</div>
""", unsafe_allow_html=True)

scenarios = compute_whatif(inputs, base_prob, models)

if not scenarios:
    st.success("🏆 Your profile is already maxed out on all scenarios!")
else:
    # ── Delta chart ───────────────────────────────────────────────────────────
    labels_wi  = [s[0] for s in scenarios]
    deltas     = [s[1] * 100 for s in scenarios]
    new_probs  = [s[2] * 100 for s in scenarios]

    fig_wi, ax_wi = plt.subplots(figsize=(10, max(3.5, len(scenarios) * 0.8)))
    fig_wi.patch.set_facecolor("#13161e"); ax_wi.set_facecolor("#0d0f14")

    bar_colors = ["#22c55e" if d > 0 else "#ef4444" for d in deltas]
    bars_wi = ax_wi.barh(labels_wi, deltas, color=bar_colors,
                          height=0.55, edgecolor="#0d0f14", zorder=3)
    for bar, delta, new_p in zip(bars_wi, deltas, new_probs):
        sign = "+" if delta >= 0 else ""
        ax_wi.text(
            delta + (0.15 if delta >= 0 else -0.15),
            bar.get_y() + bar.get_height()/2,
            f"{sign}{delta:.1f}pp  →  {new_p:.1f}%",
            va="center", ha="left" if delta >= 0 else "right",
            color="white", fontsize=9, fontweight="bold"
        )
    ax_wi.axvline(0, color="#374151", linewidth=1.5, zorder=2)
    ax_wi.set_xlabel("Change in Placement Probability (pp = percentage points)",
                      color="#6b7280", fontsize=9)
    ax_wi.tick_params(colors="#9ca3af", labelsize=9.5)
    ax_wi.spines[:].set_color("#1f2330")
    ax_wi.grid(axis="x", color="#1f2330", linewidth=0.5, zorder=0)
    ax_wi.set_title("Impact of Each Single Improvement on Placement Probability",
                    color="#9ca3af", fontsize=11, pad=10)
    plt.tight_layout()
    st.pyplot(fig_wi, use_container_width=True)
    plt.close(fig_wi)

    # ── Scenario cards ────────────────────────────────────────────────────────
    st.markdown(section_head("Scenario Breakdown"), unsafe_allow_html=True)
    for label, delta, new_prob in scenarios:
        delta_pct = delta * 100
        sign  = "+" if delta_pct >= 0 else ""
        color = "#22c55e" if delta_pct > 1 else ("#f59e0b" if delta_pct > 0 else "#ef4444")
        st.markdown(f"""
        <div style='background:#13161e;border:1px solid #1f2330;border-radius:12px;
                    padding:1.1rem 1.4rem;margin-bottom:0.6rem;
                    transition:transform 0.2s;'>
          <div style='display:flex;justify-content:space-between;align-items:center;'>
            <span style='font-size:0.95rem;color:#c7caff;font-weight:600;'>{label}</span>
            <span style='font-family:Syne,sans-serif;font-size:1.15rem;
                         font-weight:800;color:{color};'>{sign}{delta_pct:.1f}pp</span>
          </div>
          <div style='font-size:0.82rem;color:#6b7280;margin-top:0.25rem;'>
            New probability: <b style='color:#9ca3af;'>{new_prob*100:.1f}%</b>
          </div>
          <div class='bar-bg' style='margin-top:0.5rem;'>
            <div class='bar-fill' style='width:{new_prob*100:.1f}%;background:{color};'></div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Combined scenario ─────────────────────────────────────────────────────
    st.markdown(section_head("⚡ Combined Best-Case Scenario"), unsafe_allow_html=True)
    top3 = [s for s in scenarios if s[1] > 0][:3]
    if top3:
        combined_inp = dict(inputs)
        for label, _, _ in top3:
            if "CGPA" in label:
                combined_inp["cgpa"] = min(10.0, combined_inp["cgpa"] + 0.5)
            elif "Internship" in label:
                combined_inp["internships"] = min(3, combined_inp["internships"] + 1)
            elif "Project" in label:
                combined_inp["projects"] = min(6, combined_inp["projects"] + 1)
            elif "DSA" in label:
                combined_inp["dsa_skill"] = 1
                combined_inp["skill_score"] = min(4, combined_inp["skill_score"] + 1)
            elif "Python" in label:
                combined_inp["python_skill"] = 1
                combined_inp["skill_score"] = min(4, combined_inp["skill_score"] + 1)
            elif "ML" in label:
                combined_inp["ml_skill"] = 1
                combined_inp["skill_score"] = min(4, combined_inp["skill_score"] + 1)
            elif "Coding" in label:
                combined_inp["coding_score"] = min(100, combined_inp["coding_score"] + 15)
            elif "Resume" in label:
                combined_inp["resume_score"] = min(100, combined_inp["resume_score"] + 10)
            elif "Backlog" in label:
                combined_inp["backlogs"] = 0
            elif "Aptitude" in label:
                combined_inp["aptitude_score"] = min(100, combined_inp["aptitude_score"] + 10)
        combined_res = predict(combined_inp, models)
        combined_delta = (combined_res["prob"] - base_prob) * 100
        combined_color = "#22c55e" if combined_delta > 0 else "#ef4444"
        actions = " + ".join([s[0] for s in top3])
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#0a1a0a,#0d1f0d);
                    border:1px solid #22c55e;border-radius:14px;
                    padding:1.4rem 1.8rem;'>
          <div style='font-family:Syne,sans-serif;font-size:0.85rem;
                       font-weight:700;color:#22c55e;margin-bottom:0.5rem;
                       letter-spacing:0.05em;text-transform:uppercase;'>Combined Scenario</div>
          <div style='font-size:0.9rem;color:#9ca3af;margin-bottom:0.8rem;'>{actions}</div>
          <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:white;'>
            {combined_res['prob']*100:.1f}%
            <span style='font-size:1rem;color:{combined_color};margin-left:0.5rem;'>
              (+{combined_delta:.1f}pp)
            </span>
          </div>
          <div style='font-size:0.82rem;color:#6b7280;margin-top:0.3rem;'>
            Salary estimate: Rs{combined_res['salary']:.1f} LPA
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:0.78rem;color:#374151;text-align:center;margin-top:1.2rem;'>
      💡 These are model estimates assuming one change at a time. Real-world improvements may compound.
    </div>""", unsafe_allow_html=True)
