"""
pages/6_📚_Interview_Prep.py — Interview Preparation Hub
"""
import sys, pathlib, warnings
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import streamlit as st
import time

st.set_page_config(
    page_title="Interview Prep · SCP",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.helpers import (
    inject_css, load_models, predict, sidebar_profile_form,
    section_head, ROLE_ICONS, ROLE_COLORS,
)

inject_css()

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
             letter-spacing:0.18em;text-transform:uppercase;color:#22c55e;margin:0 0 0.5rem;'>Campus Placement Prep</p>
  <h1 style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#fff;margin:0;'>📚 Interview Prep Hub</h1>
  <p style='color:#6b7280;font-size:0.9rem;margin:0.6rem 0 0;'>
    Role-specific question banks, DSA checklists, and readiness assessment — all in one place.
  </p>
</div>
""", unsafe_allow_html=True)

if not MODELS_READY:
    st.error("Models not loaded.")
    st.stop()

if "result" not in st.session_state or predict_btn:
    with st.spinner("Loading…"):
        if predict_btn: time.sleep(0.3)
        st.session_state.result = predict(inputs, models)

res = st.session_state.result
role = res["role"]
role_color = ROLE_COLORS.get(role, "#7c83f5")
role_icon  = ROLE_ICONS.get(role, "💼")

# ── Interview Readiness Score ─────────────────────────────────────────────────
interview_score = min(100, (
    inputs["dsa_skill"]    * 25 +
    inputs["python_skill"] * 20 +
    inputs["coding_score"] * 0.30 +
    (inputs["communication_score"] - 4) / 6 * 15 +
    min(inputs["internships"] / 2, 1) * 10
))
iw_level = "Interview-Ready 🟢" if interview_score >= 75 else ("Almost Ready 🟡" if interview_score >= 55 else "Needs Prep 🔴")
iw_color  = "#22c55e" if interview_score >= 75 else ("#f59e0b" if interview_score >= 55 else "#ef4444")

rs1, rs2, rs3 = st.columns(3)
with rs1:
    st.markdown(f"""
    <div class='card card-placed'>
      <div class='card-label'>Interview Readiness</div>
      <div class='card-value' style='color:{iw_color};'>{interview_score:.0f} / 100</div>
      <div class='card-sub'>{iw_level}</div>
      <div class='bar-bg'><div class='bar-fill' style='width:{interview_score:.0f}%;background:{iw_color};'></div></div>
    </div>""", unsafe_allow_html=True)
with rs2:
    st.markdown(f"""
    <div class='card card-role'>
      <div class='card-label'>Predicted Role</div>
      <div class='card-value'>{role_icon} {role}</div>
      <div class='card-sub'>Interview questions customized for this role</div>
    </div>""", unsafe_allow_html=True)
with rs3:
    days_needed = max(7, int((100 - interview_score) * 0.8))
    st.markdown(f"""
    <div class='card card-salary'>
      <div class='card-label'>Estimated Prep Time</div>
      <div class='card-value'>{days_needed} Days</div>
      <div class='card-sub'>To reach interview-ready status</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
it1, it2, it3, it4 = st.tabs([
    "  💬  Role Questions  ",
    "  🧩  DSA Checklist  ",
    "  🏢  Company Tiers  ",
    "  🧪  Mock Assessment  ",
])

# ── Tab 1: Role Questions ─────────────────────────────────────────────────────
with it1:
    QUESTIONS = {
        "Data Scientist": {
            "Technical": [
                "Explain the bias-variance tradeoff and how you manage it in practice.",
                "What is the difference between bagging and boosting?",
                "How does gradient descent work? What's the difference between SGD, Adam, and RMSProp?",
                "Explain precision, recall, F1, and AUC-ROC. When would you use each?",
                "What is regularization (L1 vs L2)? How does it prevent overfitting?",
                "Explain how Random Forest selects features for each tree.",
                "What is cross-validation? Why use K-fold vs leave-one-out?",
                "How would you handle missing data in a dataset with 30% nulls?",
                "Explain the curse of dimensionality and how PCA addresses it.",
                "What is the difference between supervised, unsupervised, and semi-supervised learning?",
            ],
            "SQL / Analytics": [
                "Write a query to find the second highest salary from an Employee table.",
                "Explain window functions (ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD).",
                "How would you calculate 7-day rolling average of daily sales using SQL?",
                "What's the difference between INNER JOIN, LEFT JOIN, RIGHT JOIN, and FULL OUTER JOIN?",
                "How do you identify and handle duplicate rows in a large SQL table?",
            ],
            "Behavioural": [
                "Tell me about a data project where you had to clean messy, real-world data.",
                "How do you explain a complex ML model to a non-technical stakeholder?",
                "Describe a time when your model failed in production. What did you do?",
                "How do you prioritize features when building a new model?",
            ],
        },
        "Software Engineer": {
            "DSA": [
                "Implement a function to detect a cycle in a linked list.",
                "Find the longest substring without repeating characters (Sliding Window).",
                "Serialize and deserialize a binary tree.",
                "Implement LRU Cache using a HashMap + Doubly Linked List.",
                "Given a graph, find the shortest path using Dijkstra's algorithm.",
                "Solve the 0/1 Knapsack problem using Dynamic Programming.",
                "Find all permutations of a string (Backtracking).",
                "Merge K sorted linked lists.",
                "Find the median of two sorted arrays (Binary Search).",
                "Implement a Trie for autocomplete functionality.",
            ],
            "System Design": [
                "Design a URL shortener (like bit.ly). Discuss DB schema, hashing, and scalability.",
                "Design a rate limiter for an API with 10,000 req/sec.",
                "How would you design a notification system for WhatsApp?",
                "Design a distributed cache (like Redis). Discuss eviction policies.",
                "How does a load balancer work? What are round-robin, least-connections, and consistent hashing?",
            ],
            "Behavioural": [
                "Describe a time you optimized a slow function by 10x. What was your approach?",
                "How do you handle disagreements about technical decisions in a team?",
                "Tell me about a project you're most proud of. Walk me through the architecture.",
                "How do you stay up-to-date with new technologies?",
            ],
        },
        "Web Developer": {
            "Technical": [
                "What is the difference between `==` and `===` in JavaScript?",
                "Explain event bubbling and event capturing. How do you stop propagation?",
                "What is the Virtual DOM in React? How does reconciliation work?",
                "Explain the difference between `let`, `const`, and `var`.",
                "What are React hooks? Explain useState, useEffect, and useCallback.",
                "What is CORS? How do you handle it in a Node.js/Express backend?",
                "Explain how browser rendering works (Critical Rendering Path).",
                "What is the difference between REST and GraphQL APIs?",
                "How do you optimize a React app for performance?",
                "Explain CSS specificity and how the cascade works.",
            ],
            "Project-Based": [
                "Walk me through a full-stack project you built. What were the biggest challenges?",
                "How did you implement user authentication (JWT, sessions, OAuth)?",
                "How do you handle state management in large React applications?",
                "Explain how you would implement real-time features using WebSockets.",
            ],
            "Behavioural": [
                "Tell me about a time you improved the performance of a web page significantly.",
                "How do you approach responsive design and cross-browser compatibility?",
                "Describe a bug that took you the longest to fix. What was your process?",
            ],
        },
        "Analyst": {
            "Technical": [
                "Write a SQL query to calculate month-over-month growth rate for each product.",
                "What is the difference between OLAP and OLTP systems?",
                "How would you design a dashboard to track KPIs for an e-commerce company?",
                "Explain what a cohort analysis is and when you'd use it.",
                "How do you calculate Customer Lifetime Value (CLV)?",
                "What is A/B testing? How do you determine statistical significance?",
                "Walk me through a data pipeline you've built or designed.",
                "What's the difference between mean, median, and mode? When is each appropriate?",
                "How would you detect anomalies in time-series data?",
                "Explain the concept of p-value and confidence intervals in plain English.",
            ],
            "Case Studies": [
                "Our app's DAU dropped 20% last week. How would you investigate the root cause?",
                "How would you segment our users to improve email open rates?",
                "Estimate the number of daily Uber rides in Mumbai.",
                "How would you measure the success of a new product feature?",
            ],
            "Behavioural": [
                "Tell me about an insight you discovered that changed a business decision.",
                "How do you handle working with dirty or incomplete data?",
                "How do you communicate complex findings to non-technical stakeholders?",
            ],
        },
    }

    q_data = QUESTIONS.get(role, QUESTIONS["Software Engineer"])
    st.markdown(f"""
    <div style='background:#13161e;border:1px solid #1f2330;border-left:3px solid {role_color};
                border-radius:12px;padding:1rem 1.4rem;margin-bottom:1rem;'>
      <div style='font-family:Syne,sans-serif;font-size:0.9rem;font-weight:700;color:#c7caff;'>
        {role_icon} {role} — Question Bank
      </div>
      <div style='font-size:0.82rem;color:#6b7280;margin-top:0.2rem;'>
        {sum(len(v) for v in q_data.values())} curated questions across {len(q_data)} categories
      </div>
    </div>
    """, unsafe_allow_html=True)

    for category, questions in q_data.items():
        with st.expander(f"📋 {category} ({len(questions)} questions)", expanded=True):
            for i, q in enumerate(questions, 1):
                st.markdown(f"""
                <div style='background:#0d0f14;border:1px solid #1f2330;border-radius:8px;
                            padding:0.8rem 1.1rem;margin-bottom:0.4rem;'>
                  <span style='font-size:0.72rem;font-weight:700;color:#4b5563;margin-right:0.5rem;'>{i:02d}.</span>
                  <span style='font-size:0.85rem;color:#c7caff;'>{q}</span>
                </div>""", unsafe_allow_html=True)

# ── Tab 2: DSA Checklist ──────────────────────────────────────────────────────
with it2:
    st.markdown(section_head("🧩 DSA Topic Mastery Checklist"), unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.85rem;color:#6b7280;margin-bottom:1rem;'>
      Track your progress across essential DSA topics. Check off what you've mastered.
      Aim to complete all Easy + Medium topics before campus season.
    </div>""", unsafe_allow_html=True)

    DSA_TOPICS = {
        "🟢 Foundation (Must Know)": [
            ("Arrays & Hashing",       "Two Sum, Group Anagrams, Top K Frequent",      "Easy"),
            ("Two Pointers",           "Valid Palindrome, 3Sum, Container with Water",  "Easy"),
            ("Sliding Window",         "Longest Substring, Max Subarray (Kadane's)",    "Easy"),
            ("Stack",                  "Valid Parentheses, Min Stack, Daily Temperatures","Easy"),
            ("Binary Search",          "Search in Rotated Array, Find Min in Rotated", "Medium"),
            ("Linked List",            "Reverse List, Detect Cycle, Merge Two Lists",   "Easy"),
        ],
        "🟡 Intermediate (Important)": [
            ("Trees — BFS/DFS",        "Level Order, Max Depth, Same Tree",             "Medium"),
            ("Trees — Advanced",       "Serialize/Deserialize, LCA, Validate BST",      "Medium"),
            ("Heap / Priority Queue",  "K Closest Points, Find Median from Stream",     "Medium"),
            ("Backtracking",           "Subsets, Combinations, Permutations, N-Queens", "Medium"),
            ("Graphs",                 "Number of Islands, Clone Graph, Course Schedule","Medium"),
            ("Tries",                  "Implement Trie, Word Search II",                "Medium"),
        ],
        "🔴 Advanced (Differentiator)": [
            ("Dynamic Programming",    "Coin Change, LCS, Edit Distance, Knapsack",    "Hard"),
            ("Advanced Graphs",        "Dijkstra, Bellman-Ford, Union-Find, Prim's",   "Hard"),
            ("Intervals",              "Merge Intervals, Meeting Rooms II",             "Medium"),
            ("Bit Manipulation",       "Single Number, Number of 1 Bits, Reverse Bits","Medium"),
            ("Math & Geometry",        "Rotate Image, Spiral Matrix, Set Zeroes",      "Medium"),
        ],
    }

    for category, topics in DSA_TOPICS.items():
        st.markdown(f"**{category}**")
        for i, (topic, examples, level) in enumerate(topics):
            level_color = "#22c55e" if level == "Easy" else ("#f59e0b" if level == "Medium" else "#ef4444")
            checked = st.checkbox(topic, key=f"dsa_{category}_{i}")
            st.markdown(f"""
            <div style='background:{"#0a1a0a" if checked else "#0d0f14"};border:1px solid {"#22c55e" if checked else "#1f2330"};
                        border-radius:8px;padding:0.6rem 1rem;margin:-0.4rem 0 0.4rem 1.8rem;'>
              <span style='font-size:0.75rem;color:#4b5563;'>Examples: </span>
              <span style='font-size:0.75rem;color:#6b7280;'>{examples}</span>
              <span style='float:right;font-size:0.7rem;font-weight:700;color:{level_color};
                           border:1px solid {level_color};border-radius:4px;padding:0 0.4rem;'>{level}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

# ── Tab 3: Company Tiers ──────────────────────────────────────────────────────
with it3:
    st.markdown(section_head("🏢 Company Tier Readiness"), unsafe_allow_html=True)

    prob = res["prob"]
    TIERS = [
        ("Tier 1 — FAANG+", "#ef4444", "90%+",
         "Google, Amazon, Microsoft, Meta, Apple, Adobe, Uber, Airbnb",
         ["DSA (Hard)", "System Design", "Behavioral", "CS Fundamentals"],
         prob >= 0.90),
        ("Tier 2 — Product Companies", "#f59e0b", "70–90%",
         "Flipkart, Swiggy, CRED, Razorpay, Zepto, PhonePe, Dream11",
         ["DSA (Medium)", "System Design basics", "Projects review", "Behavioral"],
         prob >= 0.70),
        ("Tier 3 — Mid-size / Service", "#7c83f5", "50–70%",
         "TCS, Infosys, Wipro, HCL, Cognizant, Capgemini, Accenture",
         ["Aptitude test", "Basic DSA", "Communication", "Domain knowledge"],
         prob >= 0.50),
        ("Tier 4 — Startups", "#22c55e", "Any level",
         "Early-stage startups, SaaS companies, niche product firms",
         ["Projects portfolio", "Adaptability", "Domain skills", "Cultural fit"],
         True),
    ]

    for tier_name, color, threshold, companies, requirements, eligible in TIERS:
        border = color if eligible else "#374151"
        opacity = "1" if eligible else "0.5"
        badge = "✅ You qualify" if eligible else "❌ Gap to close"
        badge_color = "#22c55e" if eligible else "#ef4444"
        st.markdown(f"""
        <div style='background:#13161e;border:1px solid {border};border-radius:12px;
                    padding:1.2rem 1.6rem;margin-bottom:0.8rem;opacity:{opacity};'>
          <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.6rem;'>
            <div>
              <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#c7caff;'>{tier_name}</div>
              <div style='font-size:0.78rem;color:#4b5563;margin-top:0.2rem;'>Prob threshold: {threshold}</div>
            </div>
            <span style='font-size:0.78rem;font-weight:700;color:{badge_color};
                         border:1px solid {badge_color};border-radius:6px;padding:0.25rem 0.6rem;'>{badge}</span>
          </div>
          <div style='font-size:0.8rem;color:#6b7280;margin-bottom:0.6rem;'>
            <b style='color:#9ca3af;'>Companies:</b> {companies}
          </div>
          <div style='font-size:0.8rem;color:#6b7280;'>
            <b style='color:#9ca3af;'>Key Requirements:</b>
            {''.join(f"<span class='chip'>{r}</span>" for r in requirements)}
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Tab 4: Mock Assessment ────────────────────────────────────────────────────
with it4:
    st.markdown(section_head("🧪 Quick Self-Assessment Quiz"), unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.85rem;color:#6b7280;margin-bottom:1rem;'>
      Answer these 10 questions honestly to get your interview readiness score.
    </div>""", unsafe_allow_html=True)

    QUIZ = [
        ("Can you implement a Binary Search from scratch without looking it up?", 10),
        ("Can you explain what a Hash Map is and its time complexity?", 8),
        ("Have you solved 50+ LeetCode problems in the past 3 months?", 12),
        ("Can you confidently explain your most complex project in 2 minutes?", 10),
        ("Do you understand the STAR method for behavioural questions?", 6),
        ("Can you estimate time/space complexity for any function you write?", 10),
        ("Have you done at least one mock interview in the past month?", 12),
        ("Can you explain what happens when you type a URL in a browser?", 8),
        ("Do you know the basics of SQL (JOIN, GROUP BY, Window Functions)?", 8),
        ("Can you explain the difference between a process and a thread?", 6),
    ]

    total_yes = 0
    max_score = sum(w for _, w in QUIZ)

    with st.form("quiz_form"):
        for i, (question, weight) in enumerate(QUIZ, 1):
            ans = st.radio(f"**Q{i}.** {question}",
                           options=["Yes ✅", "Partially 🟡", "No ❌"],
                           horizontal=True, key=f"quiz_{i}")
            if ans == "Yes ✅":
                total_yes += weight
            elif ans == "Partially 🟡":
                total_yes += weight // 2

        submitted = st.form_submit_button("📊 Calculate My Score", use_container_width=True)

    if submitted:
        pct_score = total_yes / max_score * 100
        level = ("🟢 Interview-Ready" if pct_score >= 75 else
                 ("🟡 Almost Ready — 2–3 more weeks" if pct_score >= 50 else
                  "🔴 Needs Significant Prep — 4–6 more weeks"))
        color = "#22c55e" if pct_score >= 75 else ("#f59e0b" if pct_score >= 50 else "#ef4444")

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#13161e,#1a1d2e);
                    border:2px solid {color};border-radius:14px;
                    padding:1.6rem 2rem;margin-top:1rem;text-align:center;'>
          <div style='font-family:Syne,sans-serif;font-size:2.5rem;font-weight:800;color:{color};'>{pct_score:.0f}%</div>
          <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#c7caff;margin:0.4rem 0;'>{level}</div>
          <div style='font-size:0.85rem;color:#6b7280;'>
            Score: {total_yes}/{max_score} points
          </div>
          <div class='bar-bg' style='margin:0.8rem auto;max-width:320px;'>
            <div class='bar-fill' style='width:{pct_score:.0f}%;background:{color};'></div>
          </div>
          <div style='font-size:0.82rem;color:#4b5563;margin-top:0.6rem;'>
            Visit the <b style='color:#7c83f5;'>🗺️ Career Roadmap</b> page for a personalized action plan.
          </div>
        </div>""", unsafe_allow_html=True)
