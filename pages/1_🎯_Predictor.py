"""
pages/1_🎯_Predictor.py — Full Prediction Engine
Software Career Predictor (Production)
"""
import sys, pathlib, warnings, datetime
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
import time

st.set_page_config(
    page_title="Predictor · Software Career Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.helpers import (
    inject_css, load_models, load_dataset, predict, compute_readiness,
    get_advisor_tips, compute_whatif, result_card, section_head,
    metric_pill, advisor_card, badge_html, role_prob_bars,
    sidebar_profile_form, JOB_ROLES, ROLE_ICONS, ROLE_COLORS,
)

inject_css()

# ── Load resources ────────────────────────────────────────────────────────────
try:
    models = load_models()
    MODELS_READY = True
except Exception as e:
    MODELS_READY = False
    MODEL_ERROR = str(e)

DATASET = load_dataset()

# ── Sidebar form ──────────────────────────────────────────────────────────────
inputs, predict_btn = sidebar_profile_form()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#0f1218 0%,#0a0e1a 100%);
            border:1px solid #1f2a40;border-radius:16px;padding:1.8rem 2.4rem;
            margin-bottom:1.5rem;position:relative;overflow:hidden;'>
  <div style='position:absolute;top:-50px;right:-50px;width:200px;height:200px;
              background:radial-gradient(circle,rgba(124,131,245,0.15) 0%,transparent 70%);'></div>
  <p style='font-family:Syne,sans-serif;font-size:0.75rem;font-weight:700;
             letter-spacing:0.18em;text-transform:uppercase;color:#7c83f5;
             margin:0 0 0.5rem;'>ML-Powered · 3 Models · 9,000 Students</p>
  <h1 style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;
              color:#fff;margin:0;letter-spacing:-0.02em;'>
    🎯 Career Predictor
  </h1>
</div>
""", unsafe_allow_html=True)

if not MODELS_READY:
    st.error(f"⚠️ Models not loaded: {MODEL_ERROR}\n\nRun `python src/main.py` first.")
    st.stop()

# ── Run prediction ────────────────────────────────────────────────────────────
if "result" not in st.session_state or predict_btn:
    with st.spinner("🔮 Running 3 ML models…"):
        if predict_btn:
            time.sleep(0.5)
        st.session_state.result = predict(inputs, models)
        st.session_state.inputs = inputs
        # Keep history (last 5)
        hist = st.session_state.get("history", [])
        hist.insert(0, {"inputs": dict(inputs), "result": dict(st.session_state.result),
                        "ts": datetime.datetime.now().strftime("%H:%M:%S")})
        st.session_state.history = hist[:5]

res = st.session_state.result
readiness = compute_readiness(inputs, res["prob"])

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  📊  Results  ",
    "  🔍  Explainability  ",
    "  📈  Charts  ",
    "  🕓  History  ",
])

# ============================================================
# TAB 1 — Results
# ============================================================
with tab1:
    # ── Top 3 cards ──────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    placed_class = "card-placed" if res["placed"] else "card-rejected"
    placed_icon  = "✅" if res["placed"] else "❌"
    placed_label = "PLACED" if res["placed"] else "NOT PLACED"
    bar_col      = "#22c55e" if res["placed"] else "#ef4444"
    conf_pct     = res["prob"] * 100
    sal          = res["salary"]
    sal_tier     = "🔥 Top Earner" if sal > 90 else ("💰 Above Avg" if sal > 65 else "📦 Entry Level")
    top_role_c   = res["role_probas"].get(res["role"], 0) * 100
    role_icon    = ROLE_ICONS.get(res["role"], "💼")

    with c1:
        st.markdown(result_card(
            "Placement Prediction",
            f"{placed_icon} {placed_label}",
            f"Confidence: {conf_pct:.1f}%",
            placed_class, conf_pct, bar_col
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(result_card(
            "Predicted Job Role",
            f"{role_icon} {res['role']}",
            f"Model confidence: {top_role_c:.1f}%",
            "card-role"
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(result_card(
            "Expected Salary",
            f"₹{sal:.1f} LPA",
            sal_tier,
            "card-salary"
        ), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Badge + Profile ───────────────────────────────────────
    badge_col, prof_col = st.columns([1, 2.2])
    with badge_col:
        tier_bg = {
            "Platinum": "linear-gradient(135deg,#334155,#64748b)",
            "Gold":     "linear-gradient(135deg,#78350f,#d97706)",
            "Silver":   "linear-gradient(135deg,#334155,#64748b)",
            "Bronze":   "linear-gradient(135deg,#451a03,#92400e)",
        }.get(readiness["tier"])
        st.markdown(badge_html(
            readiness["score"], readiness["pct"],
            readiness["tier"], readiness["color"], readiness["glow"]
        ), unsafe_allow_html=True)

    with prof_col:
        st.markdown(section_head("Profile Summary"), unsafe_allow_html=True)
        pc1, pc2, pc3, pc4 = st.columns(4)
        for col, num, lbl in zip(
            [pc1,pc2,pc3,pc4],
            [f"{inputs['cgpa']:.1f}", str(inputs['internships']),
             str(inputs['projects']),  str(inputs['backlogs'])],
            ["CGPA","Internships","Projects","Backlogs"]
        ):
            with col:
                st.markdown(metric_pill(num, lbl), unsafe_allow_html=True)

        skills_on = [s for s, k in [("Python","python_skill"),("DSA","dsa_skill"),
                                     ("ML","ml_skill"),("Web Dev","web_dev_skill")]
                     if inputs[k]]
        chips = "".join(f"<span class='chip'>✔ {s}</span>" for s in skills_on)
        st.markdown(f"""
        <div style='margin-top:1rem;'>
          <span class='chip'>🏫 {inputs['branch']}</span>
          <span class='chip'>🏆 Tier {inputs['college_tier']}</span>
          {chips}
          <span class='chip'>📝 Resume {inputs['resume_score']}</span>
          <span class='chip'>💬 Comm {inputs['communication_score']}</span>
          <span class='chip'>🧮 Apt {inputs['aptitude_score']}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Role probabilities ────────────────────────────────────
    if res["role_probas"]:
        st.markdown(section_head("Job Role Probability Breakdown"), unsafe_allow_html=True)
        rp_l, rp_r = st.columns(2)
        with rp_l:
            st.markdown(role_prob_bars(res["role_probas"], res["role"]), unsafe_allow_html=True)
        with rp_r:
            # Mini donut chart
            roles_sorted = sorted(res["role_probas"].items(), key=lambda x:-x[1])
            labels_ = [r for r,_ in roles_sorted]
            vals_   = [p for _,p in roles_sorted]
            colors_ = [ROLE_COLORS.get(r,"#7c83f5") for r in labels_]
            fig_d, ax_d = plt.subplots(figsize=(3.5, 3.5))
            fig_d.patch.set_facecolor("#13161e")
            ax_d.set_facecolor("#13161e")
            wedges, _, autotexts = ax_d.pie(
                vals_, labels=None, colors=colors_,
                autopct="%1.0f%%", startangle=140,
                pctdistance=0.75,
                wedgeprops={"edgecolor":"#0d0f14","linewidth":2,"width":0.55},
                textprops={"color":"white","fontsize":9,"fontweight":"bold"}
            )
            ax_d.legend(labels_, loc="center", fontsize=8,
                        labelcolor="#9ca3af", facecolor="#13161e",
                        edgecolor="#1f2330", framealpha=0.9)
            plt.tight_layout()
            st.pyplot(fig_d, use_container_width=True)
            plt.close(fig_d)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Smart Advisor ─────────────────────────────────────────
    st.markdown(section_head("🧑🏫 Smart Career Advisor"), unsafe_allow_html=True)
    tips = get_advisor_tips(inputs, res)
    for icon, title, body in tips:
        st.markdown(advisor_card(icon, title, body), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Goal Tracker ──────────────────────────────────────────
    st.markdown(section_head("🎯 Placement Goal Tracker"), unsafe_allow_html=True)
    gt_col1, gt_col2 = st.columns([1, 2])
    with gt_col1:
        target_prob = st.slider(
            "Set your target placement probability (%)",
            min_value=50, max_value=99, value=90, step=1,
            key="goal_target"
        )
    with gt_col2:
        current_pct = res["prob"] * 100
        gap         = max(0, target_prob - current_pct)
        if gap == 0:
            st.success(f"🎉 You've already hit your {target_prob}% goal! (Current: {current_pct:.1f}%)")
        else:
            st.markdown(f"""
            <div style='background:#13161e;border:1px solid #1f2330;border-radius:12px;
                        padding:1.2rem 1.5rem;'>
              <div style='font-family:Syne,sans-serif;font-size:0.88rem;
                          font-weight:700;color:#c7caff;margin-bottom:0.6rem;'>
                Gap to Goal: <span style='color:#f59e0b;'>{gap:.1f} percentage points</span>
              </div>
              <div class='bar-bg'>
                <div class='bar-fill' style='width:{current_pct:.1f}%;background:#7c83f5;'></div>
              </div>
              <div style='display:flex;justify-content:space-between;
                          font-size:0.75rem;color:#4b5563;margin-top:0.4rem;'>
                <span>Current: {current_pct:.1f}%</span>
                <span>Target: {target_prob}%</span>
              </div>
              <div style='font-size:0.82rem;color:#6b7280;margin-top:0.8rem;'>
                💡 Try the <b style='color:#c7caff;'>What-If Simulator</b> page to find
                which improvements close this gap fastest.
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Download Report ───────────────────────────────────────
    date_str = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")
    skills_list = [s for s, k in [("Python","python_skill"),("DSA","dsa_skill"),
                                   ("ML","ml_skill"),("Web Dev","web_dev_skill")]
                   if inputs[k]]
    report_lines = [
        "="*62, "  SOFTWARE CAREER PREDICTOR — CAREER REPORT",
        f"  Generated: {date_str}", "="*62, "",
        "STUDENT PROFILE", "-"*40,
        f"  CGPA         : {inputs['cgpa']:.1f}",
        f"  Branch       : {inputs['branch']}",
        f"  College Tier : Tier {inputs['college_tier']}",
        f"  Skills       : {', '.join(skills_list) or 'None'}",
        f"  Internships  : {inputs['internships']}",
        f"  Projects     : {inputs['projects']}",
        f"  Backlogs     : {inputs['backlogs']}",
        f"  Coding Score : {inputs['coding_score']}",
        f"  Resume Score : {inputs['resume_score']}",
        f"  Aptitude     : {inputs['aptitude_score']}",
        "", "PREDICTIONS", "-"*40,
        f"  Placement    : {'PLACED' if res['placed'] else 'NOT PLACED'} ({res['prob']*100:.1f}%)",
        f"  Job Role     : {res['role']}",
        f"  Salary       : Rs {res['salary']:.1f} LPA",
        "", "READINESS", "-"*40,
        f"  Score        : {readiness['score']:.1f}/100 ({readiness['tier']} Tier)",
        f"  Percentile   : Top {100-readiness['pct']:.0f}%",
        "", "ADVISOR TIPS", "-"*40,
    ]
    for i, (_, title, body) in enumerate(tips, 1):
        report_lines += [f"  {i}. {title}", f"     {body}", ""]
    report_lines += ["="*62, "  Built with Streamlit & scikit-learn", "="*62]
    report_text = "\n".join(report_lines)

    st.download_button(
        label="📥 Download Full Career Report (.txt)",
        data=report_text.encode("utf-8"),
        file_name=f"career_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ============================================================
# TAB 2 — Explainability
# ============================================================
with tab2:
    st.markdown(section_head("🔍 AI Explainability — Why This Prediction?"), unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#13161e;border:1px solid #1f2330;border-radius:12px;
                padding:1rem 1.4rem;margin-bottom:1.2rem;font-size:0.85rem;color:#6b7280;line-height:1.6;'>
      These charts show which features had the <b style='color:#c7caff;'>most impact</b> on your prediction.
      The placement chart uses logistic regression coefficients × your feature values
      (a simplified SHAP-style decomposition). The salary chart shows Random Forest feature importances.
    </div>
    """, unsafe_allow_html=True)

    expl = res.get("explanation", {})

    ex_col1, ex_col2 = st.columns(2)

    with ex_col1:
        st.markdown(section_head("Placement Decision Drivers"), unsafe_allow_html=True)
        contribs = expl.get("placement_contribs", {})
        if contribs:
            # Clean up feature names and take top 10
            clean = {}
            for k, v in contribs.items():
                name = k.replace("remainder__","").replace("num__","").replace("cat__","")
                name = name.replace("_"," ").title()
                clean[name] = v
            top = sorted(clean.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            feat_names_ = [t[0] for t in top]
            feat_vals_  = [t[1] for t in top]

            fig_e, ax_e = plt.subplots(figsize=(5.5, 4))
            fig_e.patch.set_facecolor("#13161e")
            ax_e.set_facecolor("#0d0f14")
            colors_e = ["#22c55e" if v >= 0 else "#ef4444" for v in feat_vals_]
            bars_e = ax_e.barh(feat_names_, feat_vals_, color=colors_e,
                               edgecolor="#0d0f14", height=0.6, zorder=3)
            ax_e.axvline(0, color="#374151", linewidth=1.2, zorder=2)
            for bar, val in zip(bars_e, feat_vals_):
                sign = "+" if val >= 0 else ""
                ax_e.text(
                    val + (0.002 if val >= 0 else -0.002),
                    bar.get_y() + bar.get_height()/2,
                    f"{sign}{val:.3f}",
                    va="center", ha="left" if val >= 0 else "right",
                    color="white", fontsize=7.5, fontweight="bold"
                )
            ax_e.set_xlabel("Contribution to log-odds", color="#6b7280", fontsize=9)
            ax_e.tick_params(colors="#9ca3af", labelsize=8.5)
            ax_e.spines[:].set_color("#1f2330")
            ax_e.grid(axis="x", color="#1f2330", linewidth=0.5, zorder=0)
            ax_e.set_title("Feature Contributions (Placement)",
                           color="#9ca3af", fontsize=10, pad=8)
            plt.tight_layout()
            st.pyplot(fig_e, use_container_width=True)
            plt.close(fig_e)

            # Top positive and negative
            positives = [(n, v) for n, v in top if v > 0][:3]
            negatives = [(n, v) for n, v in top if v < 0][:3]
            if positives:
                st.markdown("**✅ Helping your placement:**")
                for n, v in positives:
                    st.markdown(f"- `{n}` +{v:.4f}")
            if negatives:
                st.markdown("**⚠️ Hurting your placement:**")
                for n, v in negatives:
                    st.markdown(f"- `{n}` {v:.4f}")
        else:
            st.info("Feature contribution data not available for this model type.")

    with ex_col2:
        st.markdown(section_head("Salary Prediction Drivers"), unsafe_allow_html=True)
        sal_imp = expl.get("salary_importances", {})
        if sal_imp:
            clean_s = {}
            for k, v in sal_imp.items():
                name = k.replace("remainder__","").replace("num__","").replace("cat__","")
                name = name.replace("_"," ").title()
                clean_s[name] = v
            top_s = sorted(clean_s.items(), key=lambda x: -x[1])[:10]
            feat_s = [t[0] for t in top_s]
            imp_s  = [t[1]*100 for t in top_s]

            fig_s2, ax_s2 = plt.subplots(figsize=(5.5, 4))
            fig_s2.patch.set_facecolor("#13161e")
            ax_s2.set_facecolor("#0d0f14")
            palette_s = plt.cm.Blues(np.linspace(0.4, 0.9, len(feat_s))[::-1])
            ax_s2.barh(feat_s, imp_s, color="#f59e0b",
                       edgecolor="#0d0f14", height=0.6, zorder=3)
            for i, (bar, val) in enumerate(zip(ax_s2.patches, imp_s)):
                ax_s2.text(
                    val + 0.2, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", color="white",
                    fontsize=7.5, fontweight="bold"
                )
            ax_s2.set_xlabel("Feature Importance (%)", color="#6b7280", fontsize=9)
            ax_s2.tick_params(colors="#9ca3af", labelsize=8.5)
            ax_s2.spines[:].set_color("#1f2330")
            ax_s2.grid(axis="x", color="#1f2330", linewidth=0.5, zorder=0)
            ax_s2.set_title("Feature Importances (Salary RF)",
                            color="#9ca3af", fontsize=10, pad=8)
            plt.tight_layout()
            st.pyplot(fig_s2, use_container_width=True)
            plt.close(fig_s2)

            st.markdown("**🔑 Top salary drivers:**")
            for n, v in top_s[:3]:
                st.markdown(f"- `{n}` — {v:.1f}% importance")
        else:
            st.info("Salary feature importance not available.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ATS Score
    st.markdown(section_head("📄 ATS Compatibility Score"), unsafe_allow_html=True)
    ats_col1, ats_col2 = st.columns(2)
    with ats_col1:
        skills_count = inputs["skill_score"]
        ats_score = min(100, (
            inputs["resume_score"] * 0.35 +
            skills_count / 4 * 25 +
            min(inputs["projects"] / 4, 1) * 20 +
            min(inputs["internships"] / 2, 1) * 20
        ))
        ats_level = "Excellent" if ats_score >= 80 else ("Good" if ats_score >= 65 else ("Fair" if ats_score >= 50 else "Needs Work"))
        ats_color = "#22c55e" if ats_score >= 80 else ("#7c83f5" if ats_score >= 65 else ("#f59e0b" if ats_score >= 50 else "#ef4444"))
        st.markdown(f"""
        <div class='card card-info'>
          <div class='card-label'>ATS Compatibility</div>
          <div class='card-value' style='color:{ats_color};'>{ats_score:.0f} / 100</div>
          <div class='card-sub'>{ats_level} — Resume will pass automated screening</div>
          <div class='bar-bg'>
            <div class='bar-fill' style='width:{ats_score:.0f}%;background:{ats_color};'></div>
          </div>
        </div>""", unsafe_allow_html=True)
    with ats_col2:
        interview_score = min(100, (
            inputs["dsa_skill"] * 25 +
            inputs["python_skill"] * 20 +
            inputs["coding_score"] * 0.30 +
            (inputs["communication_score"] - 4) / 6 * 15
        ))
        iw_level = "Interview-Ready" if interview_score >= 75 else ("Almost Ready" if interview_score >= 55 else "Needs Prep")
        iw_color = "#22c55e" if interview_score >= 75 else ("#f59e0b" if interview_score >= 55 else "#ef4444")
        st.markdown(f"""
        <div class='card card-role'>
          <div class='card-label'>Interview Readiness</div>
          <div class='card-value' style='color:{iw_color};'>{interview_score:.0f} / 100</div>
          <div class='card-sub'>{iw_level}</div>
          <div class='bar-bg'>
            <div class='bar-fill' style='width:{interview_score:.0f}%;background:{iw_color};'></div>
          </div>
        </div>""", unsafe_allow_html=True)


# ============================================================
# TAB 3 — Charts
# ============================================================
with tab3:
    st.markdown(section_head("Skill Radar vs Dataset Average"), unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        skill_labels = ["Python", "DSA", "ML", "Web Dev"]
        student_vals = [inputs["python_skill"], inputs["dsa_skill"],
                        inputs["ml_skill"], inputs["web_dev_skill"]]
        avg_vals     = [0.65, 0.56, 0.30, 0.39]
        N = len(skill_labels)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        sv = student_vals + student_vals[:1]
        av = avg_vals + avg_vals[:1]
        fig_r, ax_r = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
        fig_r.patch.set_facecolor("#13161e"); ax_r.set_facecolor("#0d0f14")
        ax_r.plot(angles, av, color="#374151", linewidth=1.5, linestyle="--", label="Dataset Avg")
        ax_r.fill(angles, av, color="#374151", alpha=0.15)
        ax_r.plot(angles, sv, color="#7c83f5", linewidth=2.5, label="You")
        ax_r.fill(angles, sv, color="#7c83f5", alpha=0.25)
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(skill_labels, color="#9ca3af", fontsize=10)
        ax_r.set_yticks([0.25,0.5,0.75,1.0]); ax_r.set_yticklabels(["","","",""])
        ax_r.grid(color="#1f2330", linewidth=0.8)
        ax_r.spines["polar"].set_color("#1f2330")
        ax_r.legend(loc="upper right", bbox_to_anchor=(1.35,1.15),
                    fontsize=9, labelcolor="#9ca3af", facecolor="#13161e", edgecolor="#1f2330")
        plt.tight_layout(); st.pyplot(fig_r, use_container_width=True); plt.close(fig_r)

    with col_r:
        score_labels = ["Coding", "Comm×10", "Aptitude", "Resume"]
        score_vals   = [inputs["coding_score"], inputs["communication_score"]*10,
                        inputs["aptitude_score"], inputs["resume_score"]]
        avg_scores   = [49.8, 69.9, 70.1, 67.8]
        fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
        fig2.patch.set_facecolor("#13161e"); ax2.set_facecolor("#0d0f14")
        x = np.arange(len(score_labels)); w = 0.35
        ax2.bar(x-w/2, avg_scores, w, color="#2a2f45", label="Dataset Avg", zorder=3)
        ax2.bar(x+w/2, score_vals,  w, color="#7c83f5", label="You",        zorder=3)
        ax2.set_xticks(x); ax2.set_xticklabels(score_labels, color="#9ca3af", fontsize=9)
        ax2.set_ylim(0,115); ax2.set_ylabel("Score", color="#6b7280", fontsize=9)
        ax2.tick_params(colors="#6b7280"); ax2.spines[:].set_color("#1f2330")
        ax2.grid(axis="y", color="#1f2330", linewidth=0.6, zorder=0)
        ax2.legend(fontsize=9, labelcolor="#9ca3af", facecolor="#13161e", edgecolor="#1f2330")
        ax2.set_title("Score Comparison vs Dataset", color="#9ca3af", fontsize=10, pad=10)
        plt.tight_layout(); st.pyplot(fig2, use_container_width=True); plt.close(fig2)

    # Gauge
    st.markdown(section_head("Placement Probability Gauge"), unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(7, 3.5))
    fig3.patch.set_facecolor("#13161e"); ax3.set_facecolor("#13161e")
    prob_val = res["prob"]
    theta = np.linspace(np.pi, 0, 300)
    ax3.plot(np.cos(theta), np.sin(theta), color="#1f2330", linewidth=18, solid_capstyle="round")
    fill_theta = np.linspace(np.pi, np.pi - prob_val*np.pi, 300)
    g_col = "#22c55e" if prob_val>0.7 else ("#f59e0b" if prob_val>0.4 else "#ef4444")
    ax3.plot(np.cos(fill_theta), np.sin(fill_theta), color=g_col, linewidth=18, solid_capstyle="round")
    angle_ = np.pi - prob_val * np.pi
    ax3.annotate("", xy=(0.65*np.cos(angle_), 0.65*np.sin(angle_)), xytext=(0,0),
                 arrowprops=dict(arrowstyle="-|>", color="white", lw=2, mutation_scale=18))
    ax3.text(0, -0.22, f"{prob_val*100:.1f}%", ha="center", va="center",
             fontsize=28, fontweight="bold", color="white")
    ax3.text(0, -0.48, "Placement Probability", ha="center", color="#6b7280", fontsize=10)
    ax3.text(-1.05,-0.08,"0%", ha="center", color="#6b7280", fontsize=9)
    ax3.text(1.05,-0.08,"100%", ha="center", color="#6b7280", fontsize=9)
    ax3.set_xlim(-1.2,1.2); ax3.set_ylim(-0.65,1.05); ax3.axis("off")
    plt.tight_layout(); st.pyplot(fig3, use_container_width=True); plt.close(fig3)

    # Salary benchmark
    st.markdown(section_head("Salary Benchmark"), unsafe_allow_html=True)
    sal_benchmarks = {"Entry (P25)":49.0,"Median (P50)":65.9,
                      "Your Estimate":res["salary"],"Top (P75)":77.5,"Elite (P90)":90.0}
    fig4, ax4 = plt.subplots(figsize=(9, 2.8))
    fig4.patch.set_facecolor("#13161e"); ax4.set_facecolor("#13161e")
    labels4 = list(sal_benchmarks.keys()); vals4 = list(sal_benchmarks.values())
    colors4 = ["#374151","#4b5563","#7c83f5","#6b7280","#4b5563"]
    bars4 = ax4.barh(labels4, vals4, color=colors4, edgecolor="#0d0f14", height=0.55)
    for b4, v4 in zip(bars4, vals4):
        ax4.text(v4+0.5, b4.get_y()+b4.get_height()/2, f"Rs{v4:.1f}",
                 va="center", color="white", fontsize=9, fontweight="bold")
    ax4.set_xlim(0,105); ax4.set_xlabel("Salary (LPA)", color="#6b7280", fontsize=9)
    ax4.tick_params(colors="#9ca3af", labelsize=9); ax4.spines[:].set_color("#1f2330")
    ax4.grid(axis="x", color="#1f2330", linewidth=0.6)
    ax4.set_title("Your Salary vs Dataset Percentiles", color="#9ca3af", fontsize=10, pad=8)
    plt.tight_layout(); st.pyplot(fig4, use_container_width=True); plt.close(fig4)


# ============================================================
# TAB 4 — History
# ============================================================
with tab4:
    st.markdown(section_head("🕓 Prediction History (This Session)"), unsafe_allow_html=True)
    history = st.session_state.get("history", [])
    if not history:
        st.info("No predictions yet. Run a prediction to see history here.")
    else:
        for i, entry in enumerate(history):
            r = entry["result"]; inp_ = entry["inputs"]
            placed_str = "✅ PLACED" if r["placed"] else "❌ NOT PLACED"
            border = "#22c55e" if r["placed"] else "#ef4444"
            is_latest = i == 0
            st.markdown(f"""
            <div style='background:#13161e;border:1px solid {border};
                        border-radius:12px;padding:1rem 1.4rem;
                        margin-bottom:0.7rem;opacity:{'1' if is_latest else '0.7'};'>
              <div style='display:flex;justify-content:space-between;align-items:center;
                          margin-bottom:0.6rem;'>
                <span style='font-family:Syne,sans-serif;font-size:0.92rem;
                             font-weight:700;color:#c7caff;'>
                  {'🆕 Latest — ' if is_latest else ''}{entry['ts']}
                </span>
                <span style='font-family:Syne,sans-serif;font-size:1rem;
                             font-weight:800;color:{border};'>{placed_str}</span>
              </div>
              <div style='display:flex;gap:1.5rem;font-size:0.83rem;color:#6b7280;flex-wrap:wrap;'>
                <span>📚 CGPA <b style='color:#c7caff;'>{inp_['cgpa']:.1f}</b></span>
                <span>🎯 Prob <b style='color:#7c83f5;'>{r['prob']*100:.1f}%</b></span>
                <span>💼 Role <b style='color:#c7caff;'>{r['role']}</b></span>
                <span>💰 Salary <b style='color:#f59e0b;'>Rs{r['salary']:.1f}L</b></span>
                <span>🏢 Internships <b style='color:#c7caff;'>{inp_['internships']}</b></span>
                <span>🏫 {inp_['branch']} Tier {inp_['college_tier']}</span>
              </div>
            </div>""", unsafe_allow_html=True)

        if len(history) >= 2:
            st.markdown(section_head("📊 Session Delta (Latest vs Previous)"), unsafe_allow_html=True)
            curr = history[0]["result"]; prev = history[1]["result"]
            delta_prob = (curr["prob"] - prev["prob"]) * 100
            delta_sal  = curr["salary"] - prev["salary"]
            d1, d2, d3 = st.columns(3)
            for col, label, delta, unit in [
                (d1, "Placement Δ", delta_prob, "pp"),
                (d2, "Salary Δ",   delta_sal, " LPA"),
            ]:
                col.metric(label, f"{delta:+.1f}{unit}")
