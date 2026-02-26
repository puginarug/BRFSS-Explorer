"""
dashboard/app.py
────────────────
ScreenMind EDA Dashboard — explore how each risk factor relates to
mental health burden or life satisfaction.

Run from project root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_COLS = [
    "PHYSHLTH", "GENHLTH", "ADDEPEV3", "LSATISFY", "EMTSUPRT",
    "SDLONELY", "SDHBILLS", "EXERANY2", "SMOKE100", "SEXVAR",
    "EMPLOY1", "INCOME3", "EDUCA", "_AGEG5YR", "_BMI5CAT",
]

FEATURE_LABELS = {
    "PHYSHLTH": "Physical Health (bad days/month)",
    "GENHLTH":  "General Health (1=Excellent → 5=Poor)",
    "ADDEPEV3": "Depression Diagnosis (0=No, 1=Yes)",
    "LSATISFY": "Life Satisfaction (1=Very satisfied → 4=Dissatisfied)",
    "EMTSUPRT": "Emotional Support (1=Always → 5=Never)",
    "SDLONELY": "Loneliness (1=Always → 5=Never)",
    "SDHBILLS": "Hard to Pay Bills (0=No, 1=Yes)",
    "EXERANY2": "Exercises Regularly (0=No, 1=Yes)",
    "SMOKE100": "Smoked ≥100 Cigarettes (0=No, 1=Yes)",
    "SEXVAR":   "Sex (0=Female, 1=Male)",
    "EMPLOY1":  "Employment Status (1–8 categories)",
    "INCOME3":  "Household Income (1–11 scale)",
    "EDUCA":    "Education Level (1–6)",
    "_AGEG5YR": "Age Group (1=18–24 → 13=80+)",
    "_BMI5CAT": "BMI Category (1=Underweight → 4=Obese)",
}

FEATURE_DESCRIPTIONS = {
    "PHYSHLTH": "Days in the past 30 where physical health was not good. Strong co-morbidity with mental health burden.",
    "GENHLTH":  "Self-rated overall health on a 1–5 scale. Captures perceived wellbeing beyond any single diagnosis.",
    "ADDEPEV3": "Whether a doctor ever diagnosed the respondent with a depressive disorder.",
    "LSATISFY": "Overall life satisfaction (1=Very Satisfied → 4=Very Dissatisfied).",
    "EMTSUPRT": "How often emotional support is available when needed (1=Always → 5=Never).",
    "SDLONELY": "How often the respondent feels lonely (1=Always → 5=Never). Shows the clearest gradient of any social variable.",
    "SDHBILLS": "Whether bills are difficult to pay — a social determinant of health.",
    "EXERANY2": "Any physical activity outside of regular job in the past 30 days.",
    "SMOKE100": "Lifetime smoking history (≥100 cigarettes ever smoked).",
    "SEXVAR":   "Biological sex as recorded in the survey (0=Female, 1=Male).",
    "EMPLOY1":  "Current employment situation across 8 categories (employed, unemployed, retired, unable to work, etc.).",
    "INCOME3":  "Annual household income on an 11-level scale.",
    "EDUCA":    "Highest level of education completed (1–6).",
    "_AGEG5YR": "Age group in 5-year bands (13 groups, 18–24 up to 80+).",
    "_BMI5CAT": "CDC-computed BMI category (1=Underweight, 2=Normal, 3=Overweight, 4=Obese).",
}

# Human-readable labels for ordinal levels (used on chart x-axis)
LEVEL_LABELS: dict[str, dict[int, str]] = {
    "SDLONELY": {1: "Always lonely", 2: "Usually", 3: "Sometimes", 4: "Rarely", 5: "Never lonely"},
    "EMTSUPRT": {1: "Always supported", 2: "Usually", 3: "Sometimes", 4: "Rarely", 5: "Never supported"},
    "LSATISFY": {1: "Very Satisfied", 2: "Satisfied", 3: "Dissatisfied", 4: "Very Dissatisfied"},
    "ADDEPEV3": {0: "No depression diagnosis", 1: "Depression diagnosed"},
    "SDHBILLS": {0: "Bills manageable", 1: "Hard to pay bills"},
    "EXERANY2": {0: "No exercise", 1: "Exercises"},
    "SMOKE100": {0: "Never heavy smoker", 1: "Smoked ≥100 cigs"},
    "SEXVAR":   {0: "Female", 1: "Male"},
    "GENHLTH":  {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"},
    "_BMI5CAT": {1: "Underweight", 2: "Normal", 3: "Overweight", 4: "Obese"},
}

# PHYSHLTH bins: continuous 0-30 → 6 labelled ranges
PHYSHLTH_BINS   = [0, 1, 6, 11, 21, 31]
PHYSHLTH_LABELS = ["0 days", "1–5 days", "6–10 days", "11–20 days", "21–30 days"]

C_ACCENT = "#F0A500"   # amber — bars


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_arrays() -> dict:
    return {
        "y_reg_test": np.load(DATA_DIR / "y_reg_test.npy"),
        "X_test":     np.load(DATA_DIR / "X_test.npy"),
    }


@st.cache_resource
def load_scaler():
    return joblib.load(DATA_DIR / "scaler.pkl")


@st.cache_data
def get_original_scale_X() -> np.ndarray:
    """Inverse-transform X_test back to original (human-readable) scale."""
    return load_scaler().inverse_transform(load_arrays()["X_test"])


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _bar_chart(
    x_labels: list[str],
    y_means: list[float],
    y_counts: list[int],
    y_axis_label: str,
    title: str,
) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=x_labels,
        y=y_means,
        marker_color=C_ACCENT,
        marker_line_width=0,
        text=[f"{m:.2f}" for m in y_means],
        textposition="outside",
        textfont=dict(color="white", size=13),
        hovertemplate="<b>%{x}</b><br>Mean: %{y:.2f}<br>n=%{customdata:,}<extra></extra>",
        customdata=y_counts,
    ))
    fig.update_layout(
        title=title,
        yaxis_title=y_axis_label,
        yaxis_range=[0, max(y_means) * 1.3] if y_means else [0, 1],
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=460,
        margin=dict(t=60, b=40),
        font=dict(size=13),
    )
    return fig


def build_chart(
    x_col: str,
    y_values: np.ndarray,
    X_orig: np.ndarray,
    y_axis_label: str,
) -> tuple[go.Figure, list[tuple[str, int, float, float]]]:
    """
    Build a bar chart of mean(y) per level of x_col.
    Returns the figure and a list of (label, n, mean, std) rows for the summary table.
    """
    feat_idx = FEATURE_COLS.index(x_col)
    x_raw    = X_orig[:, feat_idx]

    if x_col == "PHYSHLTH":
        # Bin continuous 0-30 into ranges
        bin_idx = np.digitize(x_raw, PHYSHLTH_BINS) - 1
        bin_idx = np.clip(bin_idx, 0, len(PHYSHLTH_LABELS) - 1)
        levels  = list(range(len(PHYSHLTH_LABELS)))
        label_map = {i: PHYSHLTH_LABELS[i] for i in levels}
    else:
        x_int     = np.round(x_raw).astype(int)
        bin_idx   = x_int
        levels    = sorted(set(x_int))
        label_map = LEVEL_LABELS.get(x_col, {})

    x_labels, y_means, y_counts, table_rows = [], [], [], []

    for lv in levels:
        mask = bin_idx == lv
        if mask.sum() < 20:
            continue
        subset  = y_values[mask]
        label   = label_map.get(lv, str(lv))
        mean_y  = float(subset.mean())
        std_y   = float(subset.std())
        n       = int(mask.sum())

        x_labels.append(label)
        y_means.append(mean_y)
        y_counts.append(n)
        table_rows.append((label, n, mean_y, std_y))

    gradient = max(y_means) - min(y_means) if y_means else 0
    title    = (
        f"Mean {y_axis_label.split('(')[0].strip()} by {FEATURE_LABELS[x_col]}"
        f"   <sup>range: {gradient:.2f}</sup>"
    )

    fig = _bar_chart(x_labels, y_means, y_counts, y_axis_label, title)
    return fig, table_rows


# ── Page layout ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ScreenMind — Mental Health EDA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .stat-card {
      background: #1E2127;
      border-radius: 10px;
      padding: 18px 14px;
      text-align: center;
      border: 1px solid #2D3139;
  }
  .stat-value { font-size: 2.2rem; font-weight: 700; color: #F0A500; }
  .stat-label { font-size: 0.84rem; color: #9BA3AF; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Mental Health Risk — Data Explorer")
st.markdown(
    "BRFSS 2024 · 449,000 US adults · CDC telephone survey  \n"
    "Select any risk factor below to see how it relates to mental health burden or life satisfaction."
)
st.divider()

# ── Stat cards ────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.markdown('<div class="stat-card"><div class="stat-value">449k</div><div class="stat-label">Adults surveyed</div></div>', unsafe_allow_html=True)
c2.markdown('<div class="stat-card"><div class="stat-value">15</div><div class="stat-label">Risk factor features</div></div>', unsafe_allow_html=True)
c3.markdown('<div class="stat-card"><div class="stat-value">13.2%</div><div class="stat-label">Classified high-risk (&gt;14 bad days/mo)</div></div>', unsafe_allow_html=True)
c4.markdown('<div class="stat-card"><div class="stat-value">60%</div><div class="stat-label">Report zero bad mental health days</div></div>', unsafe_allow_html=True)

st.markdown("")

# ── Controls ──────────────────────────────────────────────────────────────────

ctrl_left, ctrl_right = st.columns([1, 3])

with ctrl_left:
    y_choice = st.radio(
        "Y-axis (outcome)",
        options=["Mental Health Days", "Life Satisfaction"],
        help=(
            "Mental Health Days = MENTHLTH (0–30, days/month mental health was not good).  \n"
            "Life Satisfaction = LSATISFY (1=Very Satisfied → 4=Very Dissatisfied)."
        ),
    )

# Build X options — exclude the feature that is being used as Y
x_options = [c for c in FEATURE_COLS if not (y_choice == "Life Satisfaction" and c == "LSATISFY")]

with ctrl_right:
    x_col = st.selectbox(
        "X-axis (risk factor)",
        options=x_options,
        index=x_options.index("SDLONELY"),   # default to loneliness — the key finding
        format_func=lambda col: FEATURE_LABELS[col],
    )

# ── Load data & resolve Y ─────────────────────────────────────────────────────

arrays = load_arrays()
X_orig = get_original_scale_X()

if y_choice == "Mental Health Days":
    y_values     = arrays["y_reg_test"]
    y_axis_label = "Mean Bad Mental Health Days / Month"
else:
    # LSATISFY is feature index 3
    lsatisfy_idx = FEATURE_COLS.index("LSATISFY")
    y_values     = np.round(X_orig[:, lsatisfy_idx]).astype(float)
    y_axis_label = "Mean Life Satisfaction (1=Very Satisfied → 4=Dissatisfied)"

# ── Chart ─────────────────────────────────────────────────────────────────────

fig, table_rows = build_chart(x_col, y_values, X_orig, y_axis_label)
st.plotly_chart(fig, use_container_width=True)

# ── Stats below the chart ─────────────────────────────────────────────────────

detail_left, detail_right = st.columns([2, 1])

with detail_left:
    rho, pval = stats.spearmanr(X_orig[:, FEATURE_COLS.index(x_col)], y_values)
    direction = "↑ higher feature → more burden" if rho > 0 else "↑ higher feature → less burden"
    st.metric(
        label=f"Spearman correlation with {y_axis_label.split('(')[0].strip()}",
        value=f"ρ = {rho:+.3f}",
        delta=direction,
        delta_color="inverse" if rho > 0 else "normal",
    )
    st.info(f"**{FEATURE_LABELS[x_col]}** — {FEATURE_DESCRIPTIONS[x_col]}")

with detail_right:
    import pandas as pd
    df_summary = pd.DataFrame(table_rows, columns=["Level", "n", "Mean", "Std"])
    df_summary["n"] = df_summary["n"].map("{:,}".format)
    df_summary["Mean"] = df_summary["Mean"].map("{:.2f}".format)
    df_summary["Std"]  = df_summary["Std"].map("{:.2f}".format)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
