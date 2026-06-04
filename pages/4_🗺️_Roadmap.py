"""
pages/4_🗺️_Roadmap.py — Personalized 90-Day Career Roadmap
"""
import sys, pathlib, warnings, datetime
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import streamlit as st
import time

st.set_page_config(
    page_title="Career Roadmap · SCP",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.helpers import (
    inject_css, load_models, predict, section_head,
    sidebar_profile_form, JOB_ROLES, ROLE_ICONS, ROLE_COLORS,
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
             letter-spacing:0.18em;text-transform:uppercase;color:#7c83f5;margin:0 0 0.5rem;'>Personalized</p>
  <h1 style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#fff;margin:0;'>🗺️ 90-Day Career Roadmap</h1>
  <p style='color:#6b7280;font-size:0.9rem;margin:0.6rem 0 0;'>
    A personalized week-by-week action plan to maximize your placement chances and reach your target role.
  </p>
</div>
""", unsafe_allow_html=True)

if not MODELS_READY:
    st.error(f"Models not loaded: {MODEL_ERROR}")
    st.stop()

if "result" not in st.session_state or predict_btn:
    with st.spinner("Generating roadmap…"):
        if predict_btn: time.sleep(0.4)
        st.session_state.result = predict(inputs, models)

res = st.session_state.result

# ── Roadmap data by role ───────────────────────────────────────────────────────
ROADMAPS = {
    "Data Scientist": [
        ("Week 1–2", "#22c55e", "Python & Data Foundations",
         "Solidify Python (NumPy, Pandas). Complete Kaggle's 'Intro to ML' micro-course. "
         "Set up a GitHub profile with a clean README.",
         ["Python", "NumPy", "Pandas", "Kaggle"]),
        ("Week 3–4", "#22c55e", "Machine Learning Core",
         "Complete scikit-learn tutorials. Build your first end-to-end ML pipeline on a Kaggle dataset. "
         "Practice LeetCode: 5 Array + 5 String problems.",
         ["scikit-learn", "LeetCode", "Kaggle Competitions"]),
        ("Month 2 (W5-8)", "#7c83f5", "Deep Learning & Projects",
         "Start fast.ai or Coursera Deep Learning Specialization. "
         "Deploy a model to Hugging Face Spaces or Streamlit Cloud. "
         "Write a Medium article about your project.",
         ["TensorFlow/PyTorch", "Hugging Face", "Streamlit"]),
        ("Month 3 (W9-12)", "#f59e0b", "Interview Prep & Applications",
         "Solve 60+ LeetCode (Easy/Medium). Practice ML system design questions. "
         "Apply to 15+ companies. Prepare STAR stories for behavioural rounds.",
         ["LeetCode", "ML System Design", "Resume Optimization"]),
    ],
    "Software Engineer": [
        ("Week 1–2", "#22c55e", "DSA Foundation",
         "Complete arrays, strings, hashing. Do 30 LeetCode Easy problems. "
         "Practice time complexity analysis for each solution.",
         ["LeetCode", "Arrays", "Strings", "HashMaps"]),
        ("Week 3–4", "#22c55e", "Core Data Structures",
         "Study Trees, Graphs, Stacks, Queues. Implement them from scratch. "
         "Do 20 LeetCode Medium problems on these topics.",
         ["Binary Trees", "BFS/DFS", "Linked Lists"]),
        ("Month 2 (W5-8)", "#7c83f5", "Advanced Algorithms + System Design",
         "Study Dynamic Programming, Backtracking, Greedy. "
         "Learn basic system design (Load Balancer, DB Sharding, Cache). "
         "Build a full-stack project (React + Node or FastAPI + React).",
         ["Dynamic Programming", "System Design", "Full-Stack"]),
        ("Month 3 (W9-12)", "#f59e0b", "Mock Interviews & Applications",
         "Do 10 mock interviews on Pramp / interviewing.io. "
         "Apply to 20+ companies. Prepare projects for portfolio review rounds.",
         ["Mock Interviews", "Pramp", "Portfolio"]),
    ],
    "Web Developer": [
        ("Week 1–2", "#22c55e", "HTML/CSS/JS Mastery",
         "Complete freeCodeCamp Responsive Web Design. "
         "Build 3 landing pages. Deploy with Netlify/Vercel.",
         ["HTML5", "CSS3", "JavaScript", "Netlify"]),
        ("Week 3–4", "#22c55e", "React & Modern Tooling",
         "Complete React official docs + build a To-Do and Weather app. "
         "Learn Git workflows (branches, PRs, code review).",
         ["React", "Git", "npm", "Webpack"]),
        ("Month 2 (W5-8)", "#7c83f5", "Backend & Full-Stack",
         "Learn Node.js + Express OR Django REST. "
         "Build a full-stack portfolio project with user auth, CRUD, and database.",
         ["Node.js", "PostgreSQL", "REST APIs", "Docker"]),
        ("Month 3 (W9-12)", "#f59e0b", "Performance & Job Applications",
         "Optimize projects (Lighthouse 90+). Learn TypeScript basics. "
         "Apply to 25+ web dev roles. Prepare portfolio review presentation.",
         ["TypeScript", "Lighthouse", "Portfolio Review"]),
    ],
    "Analyst": [
        ("Week 1–2", "#22c55e", "Excel & SQL Foundation",
         "Complete Mode Analytics SQL Tutorial. Master pivot tables, VLOOKUP, and charts in Excel. "
         "Download a public dataset and write 20 SQL queries.",
         ["SQL", "Excel", "Google Sheets"]),
        ("Week 3–4", "#22c55e", "Python for Data Analysis",
         "Learn Pandas + Matplotlib. Replicate 3 public Kaggle notebooks. "
         "Start a personal analysis project on a topic you care about.",
         ["Pandas", "Matplotlib", "Seaborn", "Jupyter"]),
        ("Month 2 (W5-8)", "#7c83f5", "Visualization & Business Acumen",
         "Learn Tableau Public or Power BI. Build 3 dashboards on real data. "
         "Read 'Storytelling with Data' by Cole Nussbaumer Knaflic.",
         ["Tableau", "Power BI", "Data Storytelling"]),
        ("Month 3 (W9-12)", "#f59e0b", "Case Studies & Applications",
         "Practice 10 case interview questions (McKinsey-style). "
         "Apply to 20+ analyst roles. Prepare a data portfolio with Notion or GitHub.",
         ["Case Studies", "Business Cases", "Portfolio"]),
    ],
}

role = res["role"]
roadmap = ROADMAPS.get(role, ROADMAPS["Software Engineer"])
role_color = ROLE_COLORS.get(role, "#7c83f5")
role_icon  = ROLE_ICONS.get(role, "💼")

# ── Role header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:linear-gradient(135deg,#13161e,#1a1d2e);border:1px solid #2a2f45;
            border-left:4px solid {role_color};border-radius:12px;
            padding:1.2rem 1.6rem;margin-bottom:1.5rem;'>
  <div style='font-size:0.75rem;font-weight:700;letter-spacing:0.12em;
               text-transform:uppercase;color:{role_color};margin-bottom:0.3rem;'>
    Your Target Role
  </div>
  <div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;color:white;'>
    {role_icon} {role}
  </div>
  <div style='font-size:0.85rem;color:#6b7280;margin-top:0.3rem;'>
    This 90-day roadmap is customized for the {role} track.
    Placement probability: <b style='color:{role_color};'>{res['prob']*100:.1f}%</b>
    &nbsp;·&nbsp; Expected salary: <b style='color:#f59e0b;'>₹{res['salary']:.1f} LPA</b>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs: Roadmap + Gaps ──────────────────────────────────────────────────────
rt1, rt2 = st.tabs(["  🗓️  90-Day Plan  ", "  🎯  Skill Gaps & Resources  "])

with rt1:
    st.markdown(section_head(f"Your Personalized 90-Day Roadmap → {role}"), unsafe_allow_html=True)

    for i, (period, color, title, body, resources) in enumerate(roadmap):
        res_chips = "".join(f"<span class='chip'>{r}</span>" for r in resources)
        st.markdown(f"""
        <div style='display:flex;gap:1rem;margin-bottom:1rem;align-items:flex-start;'>
          <div style='display:flex;flex-direction:column;align-items:center;flex-shrink:0;'>
            <div style='width:14px;height:14px;border-radius:50%;background:{color};
                        box-shadow:0 0 10px {color};margin-top:4px;'></div>
            {'<div style="width:2px;height:calc(100% + 1rem);background:#1f2330;margin-left:6px;margin-top:4px;"></div>' if i < len(roadmap)-1 else ''}
          </div>
          <div style='background:#13161e;border:1px solid #1f2330;border-radius:12px;
                      padding:1.1rem 1.4rem;flex:1;animation:fadeSlideIn 0.4s ease both;'>
            <div style='font-family:Syne,sans-serif;font-size:0.68rem;font-weight:700;
                         letter-spacing:0.12em;text-transform:uppercase;color:{color};
                         margin-bottom:0.3rem;'>{period}</div>
            <div style='font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;
                         color:#c7caff;margin-bottom:0.5rem;'>{title}</div>
            <div style='font-size:0.83rem;color:#6b7280;line-height:1.6;margin-bottom:0.7rem;'>{body}</div>
            <div>{res_chips}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Download roadmap
    date_str = datetime.datetime.now().strftime("%B %d, %Y")
    roadmap_txt = [
        "="*60,
        f"  90-DAY CAREER ROADMAP — {role.upper()}",
        f"  Generated: {date_str}",
        "="*60, "",
        f"Profile: CGPA {inputs['cgpa']:.1f} | Branch {inputs['branch']} | Tier {inputs['college_tier']}",
        f"Placement Probability: {res['prob']*100:.1f}%",
        f"Expected Salary: Rs{res['salary']:.1f} LPA",
        "", "-"*60,
    ]
    for period, _, title, body, resources in roadmap:
        roadmap_txt += [f"\n[{period}] {title}", body, f"Resources: {', '.join(resources)}"]
    roadmap_txt += ["", "="*60, "Software Career Predictor", "="*60]

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="📥 Download 90-Day Roadmap (.txt)",
        data="\n".join(roadmap_txt).encode("utf-8"),
        file_name=f"roadmap_{role.replace(' ','_')}_{datetime.datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
        use_container_width=True,
    )

with rt2:
    st.markdown(section_head("🎯 Your Skill Gaps"), unsafe_allow_html=True)

    gaps = []
    if inputs["dsa_skill"] == 0:
        gaps.append(("🧩", "DSA", "Critical gap — tested in 95% of tech interviews",
                     "https://leetcode.com", "LeetCode →", "#ef4444"))
    if inputs["python_skill"] == 0:
        gaps.append(("🐍", "Python", "Essential for all 4 predicted roles",
                     "https://kaggle.com/learn/python", "Kaggle Python Course →", "#ef4444"))
    if inputs["ml_skill"] == 0 and role == "Data Scientist":
        gaps.append(("🤖", "Machine Learning", "Required for your predicted Data Scientist role",
                     "https://kaggle.com/learn/intro-to-machine-learning", "Kaggle ML Course →", "#f59e0b"))
    if inputs["web_dev_skill"] == 0 and role == "Web Developer":
        gaps.append(("🌐", "Web Development", "Required for your predicted Web Developer role",
                     "https://freecodecamp.org", "freeCodeCamp →", "#f59e0b"))
    if inputs["internships"] < 2:
        gaps.append(("🏢", "Internship Experience", f"Only {inputs['internships']} internship(s) — aim for 2+",
                     "https://internshala.com", "Internshala →", "#f59e0b"))
    if inputs["resume_score"] < 70:
        gaps.append(("📄", "Resume Quality", f"Score {inputs['resume_score']}/100 — below competitive threshold",
                     "https://reddit.com/r/cscareerquestions", "r/cscareerquestions →", "#f59e0b"))
    if inputs["cgpa"] < 7.5:
        gaps.append(("📚", "Academic CGPA", f"CGPA {inputs['cgpa']:.1f} — many companies require 7.0+",
                     "https://nptel.ac.in", "NPTEL Courses →", "#7c83f5"))

    if not gaps:
        st.success("🌟 No critical skill gaps detected! Your profile is strong.")
    else:
        for icon, skill, desc, link, link_text, color in gaps:
            st.markdown(f"""
            <div style='background:#13161e;border:1px solid #1f2330;border-left:3px solid {color};
                        border-radius:12px;padding:1rem 1.4rem;margin-bottom:0.6rem;'>
              <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                <div>
                  <div style='font-family:Syne,sans-serif;font-size:0.9rem;font-weight:700;
                               color:#c7caff;margin-bottom:0.3rem;'>{icon} {skill}</div>
                  <div style='font-size:0.82rem;color:#6b7280;'>{desc}</div>
                </div>
                <a href='{link}' target='_blank' style='font-size:0.8rem;color:{color};
                   text-decoration:none;font-weight:600;white-space:nowrap;margin-left:1rem;
                   padding:0.3rem 0.7rem;border:1px solid {color};border-radius:6px;'>{link_text}</a>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(section_head("📚 Role-Specific Resources"), unsafe_allow_html=True)

    RESOURCES = {
        "Data Scientist": [
            ("📘", "Kaggle Learn", "Free hands-on ML, Python, Pandas courses", "https://kaggle.com/learn"),
            ("🎓", "Coursera ML Specialization", "Andrew Ng's gold-standard ML course", "https://coursera.org/specializations/machine-learning-introduction"),
            ("📊", "Towards Data Science", "DS articles, tutorials, and project ideas", "https://towardsdatascience.com"),
            ("🤗", "Hugging Face", "Free NLP models, spaces, and datasets", "https://huggingface.co"),
        ],
        "Software Engineer": [
            ("💻", "LeetCode", "3000+ coding problems with company tags", "https://leetcode.com"),
            ("🏗️", "System Design Primer", "GitHub's comprehensive system design guide", "https://github.com/donnemartin/system-design-primer"),
            ("🎯", "NeetCode", "Curated 150 LeetCode problems + video explanations", "https://neetcode.io"),
            ("📖", "GeeksforGeeks", "DSA tutorials + company-wise interview questions", "https://geeksforgeeks.org"),
        ],
        "Web Developer": [
            ("🆓", "freeCodeCamp", "Free full-stack web development curriculum", "https://freecodecamp.org"),
            ("⚛️", "React Docs", "Official React documentation and tutorials", "https://react.dev"),
            ("🚀", "The Odin Project", "Free full-stack web development path", "https://theodinproject.com"),
            ("🌐", "MDN Web Docs", "Complete reference for HTML, CSS, JavaScript", "https://developer.mozilla.org"),
        ],
        "Analyst": [
            ("📊", "Mode SQL Tutorial", "Intermediate SQL for data analysis", "https://mode.com/sql-tutorial"),
            ("📈", "Tableau Public", "Free data visualization tool with gallery", "https://public.tableau.com"),
            ("📚", "Storytelling with Data", "Book by Cole Nussbaumer Knaflic", "https://www.storytellingwithdata.com"),
            ("🐼", "Pandas Documentation", "Official Pandas user guide", "https://pandas.pydata.org/docs"),
        ],
    }

    for icon, title, desc, link in RESOURCES.get(role, RESOURCES["Software Engineer"]):
        st.markdown(f"""
        <a href='{link}' target='_blank' style='text-decoration:none;'>
          <div style='background:#13161e;border:1px solid #1f2330;border-radius:10px;
                      padding:0.9rem 1.2rem;margin-bottom:0.5rem;display:flex;gap:0.8rem;
                      align-items:center;transition:border-color 0.2s;'>
            <span style='font-size:1.3rem;'>{icon}</span>
            <div>
              <div style='font-family:Syne,sans-serif;font-size:0.88rem;font-weight:700;color:#c7caff;'>{title}</div>
              <div style='font-size:0.78rem;color:#6b7280;'>{desc}</div>
            </div>
            <span style='margin-left:auto;color:#7c83f5;font-size:1rem;'>→</span>
          </div>
        </a>
        """, unsafe_allow_html=True)
