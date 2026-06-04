"""
eda_plots.py — Standalone EDA visualisation suite
Generates all exploratory charts and saves them to the plots/ directory.
"""

import warnings
warnings.filterwarnings("ignore")

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT_DIR / "data" / "student_placement_salary_elite_v2.csv"
PLOTS_DIR  = ROOT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

NUMERIC_COLS = [
    "cgpa", "python_skill", "dsa_skill", "ml_skill", "web_dev_skill",
    "coding_score", "communication_score", "aptitude_score",
    "internships", "projects", "backlogs", "resume_score", "skill_score",
]

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["college_tier"] = df["college_tier"].astype(str)
    return df


# ─────────────────────────────────────────────────────────────────────────────
def plot_placement_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    counts = df["placed"].value_counts()
    colors = ["#E74C3C", "#2ECC71"]
    axes[0].pie(counts, labels=["Not Placed", "Placed"], autopct="%1.1f%%",
                colors=colors, startangle=90, wedgeprops=dict(edgecolor="white"))
    axes[0].set_title("Placement Distribution")

    sns.countplot(data=df, x="placed", hue="college_tier",
                  palette="Set2", ax=axes[1])
    axes[1].set_xticklabels(["Not Placed", "Placed"])
    axes[1].set_title("Placement by College Tier")
    axes[1].set_xlabel("")
    axes[1].legend(title="Tier")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_placement_distribution.png", dpi=150)
    plt.close(fig)
    print("  eda_placement_distribution.png")


def plot_job_role_distribution(df):
    placed = df[df["placed"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Overall
    sns.countplot(data=placed, y="job_role", order=placed["job_role"].value_counts().index,
                  palette="Set3", ax=axes[0])
    axes[0].set_title("Job Role Distribution (placed)")
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel("")

    # By branch
    ct = pd.crosstab(placed["branch"], placed["job_role"])
    ct.plot(kind="bar", ax=axes[1], colormap="tab10", edgecolor="white")
    axes[1].set_title("Job Role by Branch")
    axes[1].set_xlabel("Branch")
    axes[1].set_ylabel("Count")
    axes[1].legend(title="Job Role", fontsize=8)
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_job_role_distribution.png", dpi=150)
    plt.close(fig)
    print("  eda_job_role_distribution.png")


def plot_salary_distribution(df):
    placed = df[df["placed"] == 1]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Histogram
    axes[0].hist(placed["salary_lpa"], bins=40, color="#3498DB",
                 edgecolor="white", alpha=0.85)
    axes[0].axvline(placed["salary_lpa"].mean(), color="red",
                    linestyle="--", linewidth=1.5, label=f"Mean={placed['salary_lpa'].mean():.1f}")
    axes[0].set_title("Salary Distribution (LPA)")
    axes[0].set_xlabel("Salary (LPA)")
    axes[0].legend()

    # By job role
    sns.boxplot(data=placed, x="job_role", y="salary_lpa",
                palette="Set2", ax=axes[1])
    axes[1].set_title("Salary by Job Role")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=20)

    # By college tier
    sns.violinplot(data=placed, x="college_tier", y="salary_lpa",
                   palette="pastel", ax=axes[2], inner="box")
    axes[2].set_title("Salary by College Tier")
    axes[2].set_xlabel("College Tier")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_salary_distribution.png", dpi=150)
    plt.close(fig)
    print("  eda_salary_distribution.png")


def plot_correlation_heatmap(df):
    corr_cols = NUMERIC_COLS + ["placed", "salary_lpa"]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, square=True,
                cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, pad=15)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print("  eda_correlation_heatmap.png")


def plot_feature_distributions(df):
    n_cols = 3
    n_rows = (len(NUMERIC_COLS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5))
    axes_flat = axes.flatten()

    for i, col in enumerate(NUMERIC_COLS):
        ax = axes_flat[i]
        placed     = df[df["placed"] == 1][col]
        not_placed = df[df["placed"] == 0][col]
        ax.hist(placed, bins=25, alpha=0.65, label="Placed",    color="#2ECC71", edgecolor="none")
        ax.hist(not_placed, bins=25, alpha=0.65, label="Not Placed", color="#E74C3C", edgecolor="none")
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("")
        ax.legend(fontsize=8)

    # Turn off unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Feature Distributions — Placed vs Not Placed", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  eda_feature_distributions.png")


def plot_cgpa_vs_salary(df):
    placed = df[df["placed"] == 1].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    scatter = axes[0].scatter(placed["cgpa"], placed["salary_lpa"],
                               c=placed["coding_score"], cmap="viridis",
                               alpha=0.4, s=20, edgecolors="none")
    fig.colorbar(scatter, ax=axes[0], label="Coding Score")
    axes[0].set_xlabel("CGPA")
    axes[0].set_ylabel("Salary (LPA)")
    axes[0].set_title("CGPA vs Salary (coloured by Coding Score)")

    sns.boxplot(data=placed, x="internships", y="salary_lpa",
                palette="Blues", ax=axes[1])
    axes[1].set_title("Salary vs Number of Internships")
    axes[1].set_xlabel("Internships")
    axes[1].set_ylabel("Salary (LPA)")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_cgpa_vs_salary.png", dpi=150)
    plt.close(fig)
    print("  eda_cgpa_vs_salary.png")


def plot_skill_radar(df):
    """Average skill profile for each job role (placed students)."""
    placed = df[df["placed"] == 1]
    skill_cols = ["python_skill", "dsa_skill", "ml_skill", "web_dev_skill"]
    roles = sorted(placed["job_role"].dropna().unique())
    means = placed.groupby("job_role")[skill_cols].mean()

    N = len(skill_cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    labels = [c.replace("_skill", "").replace("_", " ").title() for c in skill_cols]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for color, role in zip(colors, roles):
        vals = means.loc[role].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=2, label=role)
        ax.fill(angles, vals, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title("Average Skill Profile by Job Role", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_skill_radar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  eda_skill_radar.png")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Generating EDA plots ===")
    df = load()
    plot_placement_distribution(df)
    plot_job_role_distribution(df)
    plot_salary_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_distributions(df)
    plot_cgpa_vs_salary(df)
    plot_skill_radar(df)
    print(f"\nAll plots saved to: {PLOTS_DIR}")
