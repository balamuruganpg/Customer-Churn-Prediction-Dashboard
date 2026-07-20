"""
ChurnGuard Pro — Premium Customer Churn Prediction Dashboard
=============================================================
A production-grade Streamlit dashboard with dark glassmorphism theme,
interactive Plotly visuals, and a live ML prediction pipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# PREMIUM CSS — DARK GLASSMORPHISM THEME
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:    #0a0e1a;
    --bg-secondary:  #111827;
    --bg-card:       rgba(17, 24, 39, 0.65);
    --glass-border:  rgba(99, 102, 241, 0.15);
    --glass-shadow:  0 8px 32px rgba(0, 0, 0, 0.35);
    --text-primary:  #f1f5f9;
    --text-secondary:#94a3b8;
    --text-muted:    #64748b;
    --indigo:        #818cf8;
    --indigo-bright: #6366f1;
    --emerald:       #34d399;
    --emerald-dark:  #059669;
    --purple:        #c084fc;
    --cyan:          #22d3ee;
    --rose:          #fb7185;
    --amber:         #fbbf24;
    --radius:        16px;
    --font:          'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --mono:          'JetBrains Mono', monospace;
}

/* ── Global body ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font) !important;
}

/* Animated gradient mesh background */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 80% 60% at 10% 20%,  rgba(99,102,241,0.12), transparent),
        radial-gradient(ellipse 60% 50% at 85% 75%,  rgba(52,211,153,0.08), transparent),
        radial-gradient(ellipse 50% 40% at 50% 10%,  rgba(192,132,252,0.06), transparent);
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #111827 100%) !important;
    border-right: 1px solid var(--glass-border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] .stRadio > label {
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: var(--text-secondary) !important;
}
[data-testid="stSidebar"] .stRadio > div > label {
    padding: 10px 16px !important;
    border-radius: 10px !important;
    margin-bottom: 4px !important;
    transition: all 0.25s cubic-bezier(.4,0,.2,1) !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(99,102,241,0.1) !important;
}
[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio > div [data-checked="true"] {
    background: rgba(99,102,241,0.15) !important;
}

/* ── Header / top bar ── */
header[data-testid="stHeader"] {
    background: rgba(10,14,26,0.85) !important;
    backdrop-filter: blur(12px) !important;
    border-bottom: 1px solid var(--glass-border) !important;
}

/* ── Main container spacing ── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1280px !important;
}

/* ── Glass card ── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    box-shadow: var(--glass-shadow);
    padding: 28px 26px;
    transition: transform 0.3s cubic-bezier(.4,0,.2,1), box-shadow 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(99,102,241,0.12), var(--glass-shadow);
}

/* ── KPI card ── */
.kpi-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px) saturate(180%);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    box-shadow: var(--glass-shadow);
    padding: 24px 22px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.3s cubic-bezier(.4,0,.2,1), box-shadow 0.3s ease;
}
.kpi-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 20px 50px rgba(99,102,241,0.15), var(--glass-shadow);
}
.kpi-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: var(--radius) var(--radius) 0 0;
}
.kpi-card.indigo::before  { background: linear-gradient(90deg, var(--indigo), var(--cyan)); }
.kpi-card.emerald::before { background: linear-gradient(90deg, var(--emerald), #6ee7b7); }
.kpi-card.purple::before  { background: linear-gradient(90deg, var(--purple), var(--rose)); }
.kpi-card.cyan::before    { background: linear-gradient(90deg, var(--cyan), var(--indigo)); }
.kpi-card.rose::before    { background: linear-gradient(90deg, var(--rose), var(--amber)); }

.kpi-icon { font-size: 2rem; margin-bottom: 8px; }
.kpi-value {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, var(--text-primary), var(--indigo));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.kpi-label {
    font-size: 0.82rem; font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 6px;
}

/* ── Metric card (model performance) ── */
.metric-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px) saturate(180%);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    box-shadow: var(--glass-shadow);
    padding: 22px 18px;
    text-align: center;
    transition: transform 0.3s cubic-bezier(.4,0,.2,1), box-shadow 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 40px rgba(99,102,241,0.12), var(--glass-shadow);
}
.metric-val { font-size: 1.8rem; font-weight: 800; }
.metric-lbl {
    font-size: 0.78rem; font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 6px;
}

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 8px 28px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.5px;
}
.risk-low    { background: rgba(52,211,153,0.15); color: var(--emerald); border: 1px solid rgba(52,211,153,0.3); }
.risk-medium { background: rgba(251,191,36,0.15); color: var(--amber);  border: 1px solid rgba(251,191,36,0.3); }
.risk-high   { background: rgba(251,113,133,0.15);color: var(--rose);   border: 1px solid rgba(251,113,133,0.3); }

/* ── Probability ring ── */
.prob-ring {
    width: 180px; height: 180px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 16px;
    position: relative;
}
.prob-ring-inner {
    width: 140px; height: 140px;
    border-radius: 50%;
    background: var(--bg-secondary);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
}
.prob-pct { font-size: 2.2rem; font-weight: 800; }
.prob-lbl { font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; }

/* ── Section header ── */
.section-header {
    font-size: 1.6rem; font-weight: 800;
    margin-bottom: 4px;
    background: linear-gradient(135deg, var(--text-primary), var(--indigo));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.section-sub {
    font-size: 0.92rem;
    color: var(--text-secondary);
    margin-bottom: 24px;
}

/* ── Insight chip ── */
.insight-chip {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 0.92rem;
    color: var(--text-primary);
    line-height: 1.5;
}

/* ── Plotly chart backgrounds ── */
.stPlotlyChart {
    border-radius: var(--radius) !important;
    overflow: hidden;
}

/* ── Dataframe styling ── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius) !important;
    overflow: hidden;
    border: 1px solid var(--glass-border) !important;
}

/* ── Buttons & inputs ── */
.stButton > button {
    background: linear-gradient(135deg, var(--indigo-bright), var(--purple)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,0.45) !important;
}

.stSelectbox, .stMultiSelect, .stNumberInput, .stTextInput {
    font-family: var(--font) !important;
}
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: var(--text-secondary);
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.15) !important;
    color: var(--indigo) !important;
}

/* ── Divider ── */
hr { border-color: var(--glass-border) !important; }

/* ── Recommendation box ── */
.rec-box {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border-left: 4px solid;
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 20px 24px;
    margin-top: 16px;
}
.rec-box.low    { border-left-color: var(--emerald); }
.rec-box.medium { border-left-color: var(--amber); }
.rec-box.high   { border-left-color: var(--rose); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.5); }

/* ── Brand badge ── */
.brand-badge {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 0; margin-bottom: 20px;
}
.brand-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, var(--indigo-bright), var(--purple));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    box-shadow: 0 4px 16px rgba(99,102,241,0.35);
}
.brand-text { font-size: 1.25rem; font-weight: 800; letter-spacing: -0.5px; }
.brand-sub  { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }

/* ── Logo line ── */
.sidebar-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--glass-border), transparent);
    margin: 16px 0 20px;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# PLOTLY THEME HELPER
# ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0", size=13),
    margin=dict(l=40, r=40, t=50, b=40),
    hoverlabel=dict(
        bgcolor="#1e293b",
        font_size=13,
        font_family="Inter, sans-serif",
        bordercolor="rgba(99,102,241,0.3)",
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
    ),
    xaxis=dict(gridcolor="rgba(99,102,241,0.07)", zerolinecolor="rgba(99,102,241,0.1)"),
    yaxis=dict(gridcolor="rgba(99,102,241,0.07)", zerolinecolor="rgba(99,102,241,0.1)"),
)

COLORS = ["#818cf8", "#34d399", "#c084fc", "#22d3ee", "#fb7185", "#fbbf24", "#a78bfa", "#6ee7b7"]


# ──────────────────────────────────────────────────────────────
# DATA & MODEL LOADING
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    return df


@st.cache_resource
def load_model():
    model = joblib.load("churn_prediction_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features


@st.cache_data
def prepare_model_data(df):
    """Replicate notebook preprocessing so we can evaluate metrics."""
    data = df.copy()
    data.drop("customerID", axis=1, inplace=True)
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        data[col] = LabelEncoder().fit_transform(data[col])

    data = pd.get_dummies(
        data,
        columns=[
            "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod",
        ],
        drop_first=True,
    )

    X = data.drop("Churn", axis=1)
    y = data["Churn"]
    return X, y


# ──────────────────────────────────────────────────────────────
# LOAD EVERYTHING
# ──────────────────────────────────────────────────────────────
try:
    df = load_data()
    model, model_features = load_model()
    X_all, y_all = prepare_model_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    DATA_LOADED = True
except Exception as e:
    DATA_LOADED = False
    st.error(f"⚠️ Error loading data or model: {e}")

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-badge">
        <div class="brand-icon">🛡️</div>
        <div>
            <div class="brand-text">ChurnGuard Pro</div>
            <div class="brand-sub">Predictive Analytics</div>
        </div>
    </div>
    <div class="sidebar-divider"></div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        [
            "📊  Overview Dashboard",
            "📈  Exploratory Analysis",
            "🤖  Model Performance",
            "🔮  Live Churn Prediction",
            "🔍  Customer Explorer",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="padding:10px 0;font-size:0.78rem;color:var(--text-muted);">
            <div style="margin-bottom:8px;font-weight:600;text-transform:uppercase;letter-spacing:1px;font-size:0.7rem;">
                Dataset Info
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span>Records</span><span style="color:var(--indigo);font-weight:600;">{len(df):,}</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span>Features</span><span style="color:var(--emerald);font-weight:600;">{df.shape[1] - 1}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span>Model</span><span style="color:var(--purple);font-weight:600;">GBClassifier</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not DATA_LOADED:
    st.stop()


# ══════════════════════════════════════════════════════════════
# 1. OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "📊  Overview Dashboard":
    st.markdown('<p class="section-header">Overview Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Key performance metrics and churn distribution at a glance</p>', unsafe_allow_html=True)

    total      = len(df)
    churned    = (df["Churn"] == "Yes").sum()
    churn_rate = churned / total * 100
    avg_tenure = df["tenure"].mean()
    avg_monthly= df["MonthlyCharges"].mean()

    # High-risk: predict on all data
    X_full_aligned = X_all.reindex(columns=model_features, fill_value=0)
    probas = model.predict_proba(X_full_aligned)[:, 1]
    high_risk = (probas >= 0.7).sum()

    # ── KPI Cards ──
    k1, k2, k3, k4, k5 = st.columns(5)
    cards = [
        (k1, "👥", f"{total:,}",             "Total Customers",    "indigo"),
        (k2, "📉", f"{churn_rate:.1f}%",      "Churn Rate",         "rose"),
        (k3, "📅", f"{avg_tenure:.1f}",       "Avg Tenure (mo)",    "emerald"),
        (k4, "💰", f"${avg_monthly:.2f}",     "Avg Monthly Charges","cyan"),
        (k5, "⚠️", f"{high_risk:,}",          "High Risk",          "purple"),
    ]
    for col, icon, val, label, color in cards:
        col.markdown(f"""
        <div class="kpi-card {color}">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ──
    col_pie, col_insight = st.columns([1, 1])

    with col_pie:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        churn_counts = df["Churn"].value_counts()
        fig_pie = go.Figure(
            go.Pie(
                labels=churn_counts.index.map({"Yes": "Churned", "No": "Retained"}),
                values=churn_counts.values,
                hole=0.6,
                marker=dict(colors=["#fb7185", "#34d399"], line=dict(color="#0a0e1a", width=3)),
                textinfo="percent+label",
                textfont=dict(size=14, family="Inter"),
                hoverinfo="label+value+percent",
            )
        )
        fig_pie.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Churn Distribution", font=dict(size=16, color="#e2e8f0")),
            showlegend=False,
            height=370,
            annotations=[
                dict(
                    text=f"<b>{churn_rate:.1f}%</b><br><span style='font-size:11px;color:#94a3b8'>churned</span>",
                    x=0.5, y=0.5, font_size=22, showarrow=False, font_color="#f1f5f9",
                )
            ],
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_insight:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:1.1rem;font-weight:700;margin-bottom:16px;color:var(--text-primary);">📌 Key Business Insights</div>', unsafe_allow_html=True)

        mtm_churn  = df[df["Contract"] == "Month-to-month"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
        fiber_churn= df[df["InternetService"] == "Fiber optic"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
        echeck_churn = df[df["PaymentMethod"] == "Electronic check"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
        short_tenure = df[df["tenure"] <= 12]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
        senior_churn = df[df["SeniorCitizen"] == 1]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100

        insights = [
            f"🔴 **Month-to-month** contracts have a **{mtm_churn:.1f}%** churn rate — the highest among all contract types.",
            f"🔴 **Fiber optic** internet users churn at **{fiber_churn:.1f}%** — significantly higher than DSL or no internet.",
            f"🔴 **Electronic check** payers show **{echeck_churn:.1f}%** churn — consider promoting auto-pay methods.",
            f"🟡 Customers with **≤ 12 months tenure** churn at **{short_tenure:.1f}%** — early engagement programs are critical.",
            f"🟡 **Senior citizens** churn at **{senior_churn:.1f}%** — targeted retention may help.",
        ]
        for ins in insights:
            st.markdown(f'<div class="insight-chip">{ins}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Monthly charges distribution ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    fig_dist = go.Figure()
    for label, color in [("No", "#34d399"), ("Yes", "#fb7185")]:
        subset = df[df["Churn"] == label]["MonthlyCharges"]
        fig_dist.add_trace(go.Histogram(
            x=subset, name="Churned" if label == "Yes" else "Retained",
            marker_color=color, opacity=0.7, nbinsx=40,
        ))
    fig_dist.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Monthly Charges Distribution by Churn Status", font=dict(size=16, color="#e2e8f0")),
        barmode="overlay", height=350,
        xaxis_title="Monthly Charges ($)", yaxis_title="Count",
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 2. EXPLORATORY ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "📈  Exploratory Analysis":
    st.markdown('<p class="section-header">Exploratory Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Dive deep into the data with interactive visualizations</p>', unsafe_allow_html=True)

    # ── Filters ──
    with st.expander("🎛️  Filters", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        gender_filter   = fc1.multiselect("Gender", df["gender"].unique(), default=df["gender"].unique())
        contract_filter = fc2.multiselect("Contract", df["Contract"].unique(), default=df["Contract"].unique())
        internet_filter = fc3.multiselect("Internet Service", df["InternetService"].unique(), default=df["InternetService"].unique())

    filtered = df[
        (df["gender"].isin(gender_filter)) &
        (df["Contract"].isin(contract_filter)) &
        (df["InternetService"].isin(internet_filter))
    ]

    st.markdown(f'<div style="font-size:0.82rem;color:var(--text-muted);margin-bottom:12px;">Showing <b style="color:var(--indigo)">{len(filtered):,}</b> of {len(df):,} customers</div>', unsafe_allow_html=True)

    # Row 1: Contract & Internet Service
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        ct = filtered.groupby(["Contract", "Churn"]).size().reset_index(name="Count")
        ct["Churn"] = ct["Churn"].map({"Yes": "Churned", "No": "Retained"})
        fig_ct = px.bar(
            ct, x="Contract", y="Count", color="Churn",
            barmode="group", color_discrete_map={"Churned": "#fb7185", "Retained": "#34d399"},
            title="Churn by Contract Type",
        )
        fig_ct.update_layout(**PLOTLY_LAYOUT, height=380)
        st.plotly_chart(fig_ct, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r1c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        it = filtered.groupby(["InternetService", "Churn"]).size().reset_index(name="Count")
        it["Churn"] = it["Churn"].map({"Yes": "Churned", "No": "Retained"})
        fig_it = px.bar(
            it, x="InternetService", y="Count", color="Churn",
            barmode="group", color_discrete_map={"Churned": "#fb7185", "Retained": "#34d399"},
            title="Churn by Internet Service",
        )
        fig_it.update_layout(**PLOTLY_LAYOUT, height=380)
        st.plotly_chart(fig_it, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Payment Method & Scatter
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        pm = filtered.groupby(["PaymentMethod", "Churn"]).size().reset_index(name="Count")
        pm["Churn"] = pm["Churn"].map({"Yes": "Churned", "No": "Retained"})
        fig_pm = px.bar(
            pm, x="Count", y="PaymentMethod", color="Churn",
            barmode="group", orientation="h",
            color_discrete_map={"Churned": "#fb7185", "Retained": "#34d399"},
            title="Churn by Payment Method",
        )
        fig_pm.update_layout(**PLOTLY_LAYOUT, height=400)
        st.plotly_chart(fig_pm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r2c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        scatter_sample = filtered.sample(min(1500, len(filtered)), random_state=42)
        fig_sc = px.scatter(
            scatter_sample, x="tenure", y="MonthlyCharges", color="Churn",
            color_discrete_map={"Yes": "#fb7185", "No": "#34d399"},
            opacity=0.6, title="Tenure vs Monthly Charges",
            labels={"tenure": "Tenure (months)", "MonthlyCharges": "Monthly Charges ($)"},
        )
        fig_sc.update_layout(**PLOTLY_LAYOUT, height=400)
        fig_sc.update_traces(marker=dict(size=6, line=dict(width=0.5, color="#0a0e1a")))
        st.plotly_chart(fig_sc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 3: Tenure distribution
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    fig_ten = go.Figure()
    for label, color, name in [("No", "#34d399", "Retained"), ("Yes", "#fb7185", "Churned")]:
        fig_ten.add_trace(go.Histogram(
            x=filtered[filtered["Churn"] == label]["tenure"],
            name=name, marker_color=color, opacity=0.7, nbinsx=30,
        ))
    fig_ten.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Tenure Distribution by Churn Status", font=dict(size=16, color="#e2e8f0")),
        barmode="overlay", height=350,
        xaxis_title="Tenure (months)", yaxis_title="Count",
    )
    st.plotly_chart(fig_ten, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 3. MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "🤖  Model Performance":
    st.markdown('<p class="section-header">Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Gradient Boosting Classifier evaluation on the held-out test set</p>', unsafe_allow_html=True)

    # Predict
    X_test_aligned = X_test.reindex(columns=model_features, fill_value=0)
    y_pred  = model.predict(X_test_aligned)
    y_proba = model.predict_proba(X_test_aligned)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)

    # ── Metric cards ──
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        (m1, "Accuracy",  f"{acc:.4f}",  "#818cf8"),
        (m2, "Precision", f"{prec:.4f}", "#34d399"),
        (m3, "Recall",    f"{rec:.4f}",  "#c084fc"),
        (m4, "F1-Score",  f"{f1:.4f}",   "#22d3ee"),
        (m5, "ROC-AUC",   f"{auc:.4f}",  "#fbbf24"),
    ]
    for col, lbl, val, clr in metrics:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color:{clr};">{val}</div>
            <div class="metric-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confusion Matrix & ROC ──
    cm_col, roc_col = st.columns(2)

    with cm_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = go.Figure(go.Heatmap(
            z=cm[::-1], x=["No Churn", "Churn"], y=["Churn", "No Churn"],
            text=cm[::-1], texttemplate="%{text}",
            colorscale=[[0, "#1e1b4b"], [0.5, "#4338ca"], [1, "#818cf8"]],
            showscale=False, hoverinfo="skip",
            textfont=dict(size=20, color="white"),
        ))
        fig_cm.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Confusion Matrix", font=dict(size=16, color="#e2e8f0")),
            height=400,
            xaxis_title="Predicted", yaxis_title="Actual",
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with roc_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            line=dict(color="#818cf8", width=3),
            fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
            name=f"AUC = {auc:.4f}",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#4b5563", width=1, dash="dash"),
            name="Random", showlegend=True,
        ))
        fig_roc.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="ROC Curve", font=dict(size=16, color="#e2e8f0")),
            height=400,
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature Importance ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=model_features).nlargest(15)
        fig_fi = go.Figure(go.Bar(
            x=importances.values,
            y=importances.index,
            orientation="h",
            marker=dict(
                color=importances.values,
                colorscale=[[0, "#312e81"], [0.3, "#4338ca"], [0.6, "#818cf8"], [1, "#c084fc"]],
                line=dict(width=0),
            ),
        ))
        fig_fi.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Top 15 Feature Importances", font=dict(size=16, color="#e2e8f0")),
            height=500,
            xaxis_title="Importance Score",
        )
        fig_fi.update_yaxes(autorange="reversed", gridcolor="rgba(99,102,241,0.07)")
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importances are not available for this model type.")
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 4. LIVE CHURN PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == "🔮  Live Churn Prediction":
    st.markdown('<p class="section-header">Live Churn Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Enter customer details to predict churn probability in real time</p>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:1rem;font-weight:700;margin-bottom:18px;color:var(--text-primary);">🧾 Customer Information</div>', unsafe_allow_html=True)

        # Row 1
        c1, c2, c3, c4 = st.columns(4)
        gender           = c1.selectbox("Gender", ["Male", "Female"])
        senior           = c2.selectbox("Senior Citizen", ["No", "Yes"])
        partner          = c3.selectbox("Partner", ["No", "Yes"])
        dependents       = c4.selectbox("Dependents", ["No", "Yes"])

        # Row 2
        c5, c6, c7, c8 = st.columns(4)
        tenure           = c5.number_input("Tenure (months)", 0, 72, 12)
        monthly          = c6.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0, step=0.5)
        total_charges    = c7.number_input("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly), step=10.0)
        phone_service    = c8.selectbox("Phone Service", ["No", "Yes"])

        # Row 3
        c9, c10, c11, c12 = st.columns(4)
        multiple_lines   = c9.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet         = c10.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security  = c11.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup    = c12.selectbox("Online Backup", ["No", "Yes", "No internet service"])

        # Row 4
        c13, c14, c15, c16 = st.columns(4)
        device_protection= c13.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support     = c14.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv     = c15.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = c16.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        # Row 5
        c17, c18 = st.columns(2)
        contract         = c17.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment          = c18.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])

        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Predict Button ──
    if st.button("🔮  Predict Churn", use_container_width=True):
        # Build raw dict
        le_map = {"Male": 1, "Female": 0, "Yes": 1, "No": 0}

        input_dict = {
            "gender":           le_map[gender],
            "SeniorCitizen":    1 if senior == "Yes" else 0,
            "Partner":          le_map[partner],
            "Dependents":       le_map[dependents],
            "tenure":           tenure,
            "PhoneService":     le_map[phone_service],
            "PaperlessBilling": le_map[paperless],
            "MonthlyCharges":   monthly,
            "TotalCharges":     total_charges,
            # One-hot: MultipleLines
            "MultipleLines_No phone service": 1 if multiple_lines == "No phone service" else 0,
            "MultipleLines_Yes":              1 if multiple_lines == "Yes" else 0,
            # One-hot: InternetService
            "InternetService_Fiber optic":    1 if internet == "Fiber optic" else 0,
            "InternetService_No":             1 if internet == "No" else 0,
            # One-hot: OnlineSecurity
            "OnlineSecurity_No internet service": 1 if online_security == "No internet service" else 0,
            "OnlineSecurity_Yes":                 1 if online_security == "Yes" else 0,
            # One-hot: OnlineBackup
            "OnlineBackup_No internet service":   1 if online_backup == "No internet service" else 0,
            "OnlineBackup_Yes":                   1 if online_backup == "Yes" else 0,
            # One-hot: DeviceProtection
            "DeviceProtection_No internet service": 1 if device_protection == "No internet service" else 0,
            "DeviceProtection_Yes":                 1 if device_protection == "Yes" else 0,
            # One-hot: TechSupport
            "TechSupport_No internet service":  1 if tech_support == "No internet service" else 0,
            "TechSupport_Yes":                  1 if tech_support == "Yes" else 0,
            # One-hot: StreamingTV
            "StreamingTV_No internet service":  1 if streaming_tv == "No internet service" else 0,
            "StreamingTV_Yes":                  1 if streaming_tv == "Yes" else 0,
            # One-hot: StreamingMovies
            "StreamingMovies_No internet service": 1 if streaming_movies == "No internet service" else 0,
            "StreamingMovies_Yes":                 1 if streaming_movies == "Yes" else 0,
            # One-hot: Contract
            "Contract_One year":  1 if contract == "One year" else 0,
            "Contract_Two year":  1 if contract == "Two year" else 0,
            # One-hot: PaymentMethod
            "PaymentMethod_Credit card (automatic)": 1 if payment == "Credit card (automatic)" else 0,
            "PaymentMethod_Electronic check":        1 if payment == "Electronic check" else 0,
            "PaymentMethod_Mailed check":            1 if payment == "Mailed check" else 0,
        }

        input_df = pd.DataFrame([input_dict]).reindex(columns=model_features, fill_value=0)

        try:
            prob = model.predict_proba(input_df)[0][1]
            pct  = prob * 100

            if pct < 30:
                risk, risk_cls, ring_color = "Low Risk", "risk-low", "#34d399"
                rec_cls = "low"
                rec_msg = ("✅ This customer shows <b>strong retention signals</b>. "
                           "Continue delivering excellent service and consider loyalty rewards "
                           "to maintain engagement.")
            elif pct < 70:
                risk, risk_cls, ring_color = "Medium Risk", "risk-medium", "#fbbf24"
                rec_cls = "medium"
                rec_msg = ("⚡ This customer is at <b>moderate risk</b> of churning. "
                           "Proactive outreach is recommended — offer a contract upgrade, "
                           "a discount on add-on services, or a personalised retention call.")
            else:
                risk, risk_cls, ring_color = "High Risk", "risk-high", "#fb7185"
                rec_cls = "high"
                rec_msg = ("🚨 <b>Immediate intervention required.</b> This customer is highly likely "
                           "to churn. Deploy your best retention offer — consider a significant discount, "
                           "complimentary service upgrade, or a dedicated account manager.")

            # ── Result display ──
            st.markdown("<br>", unsafe_allow_html=True)
            res1, res2, res3 = st.columns([1, 1, 1])

            with res1:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center;">
                    <div class="prob-ring" style="background: conic-gradient({ring_color} {pct * 3.6}deg, #1e293b {pct * 3.6}deg);">
                        <div class="prob-ring-inner">
                            <div class="prob-pct" style="color:{ring_color};">{pct:.1f}%</div>
                            <div class="prob-lbl">Churn Probability</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with res2:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:230px;">
                    <div style="font-size:0.82rem;color:var(--text-secondary);text-transform:uppercase;letter-spacing:1px;margin-bottom:16px;">Risk Assessment</div>
                    <div class="risk-badge {risk_cls}" style="font-size:1.3rem;padding:14px 36px;">{risk}</div>
                    <div style="margin-top:16px;font-size:0.85rem;color:var(--text-muted);">
                        Based on {len(model_features)} predictive features
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with res3:
                st.markdown(f"""
                <div class="glass-card" style="display:flex;flex-direction:column;justify-content:center;min-height:230px;">
                    <div style="font-size:0.82rem;color:var(--text-secondary);text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">📋 Recommendation</div>
                    <div class="rec-box {rec_cls}">
                        {rec_msg}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ══════════════════════════════════════════════════════════════
# 5. CUSTOMER EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "🔍  Customer Explorer":
    st.markdown('<p class="section-header">Customer Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Search and filter customer records with ease</p>', unsafe_allow_html=True)

    # ── Search & Filters ──
    s1, s2, s3, s4 = st.columns([2, 1, 1, 1])
    search_term     = s1.text_input("🔎 Search by Customer ID", placeholder="e.g. 7590-VHVEG")
    churn_filter    = s2.selectbox("Churn Status", ["All", "Yes", "No"])
    contract_f      = s3.selectbox("Contract Type", ["All"] + list(df["Contract"].unique()))
    internet_f      = s4.selectbox("Internet Service", ["All"] + list(df["InternetService"].unique()))

    explore_df = df.copy()
    if search_term:
        explore_df = explore_df[explore_df["customerID"].str.contains(search_term, case=False, na=False)]
    if churn_filter != "All":
        explore_df = explore_df[explore_df["Churn"] == churn_filter]
    if contract_f != "All":
        explore_df = explore_df[explore_df["Contract"] == contract_f]
    if internet_f != "All":
        explore_df = explore_df[explore_df["InternetService"] == internet_f]

    st.markdown(f"""
    <div style="display:flex;gap:16px;margin-bottom:16px;">
        <div class="insight-chip" style="flex:1;text-align:center;">
            <span style="font-size:1.4rem;font-weight:800;color:var(--indigo);">{len(explore_df):,}</span>
            <span style="color:var(--text-secondary);font-size:0.85rem;margin-left:8px;">customers found</span>
        </div>
        <div class="insight-chip" style="flex:1;text-align:center;">
            <span style="font-size:1.4rem;font-weight:800;color:var(--rose);">
                {(explore_df['Churn']=='Yes').sum():,}
            </span>
            <span style="color:var(--text-secondary);font-size:0.85rem;margin-left:8px;">churned</span>
        </div>
        <div class="insight-chip" style="flex:1;text-align:center;">
            <span style="font-size:1.4rem;font-weight:800;color:var(--emerald);">
                {(explore_df['Churn']=='No').sum():,}
            </span>
            <span style="color:var(--text-secondary);font-size:0.85rem;margin-left:8px;">retained</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        explore_df,
        use_container_width=True,
        height=520,
        column_config={
            "customerID": st.column_config.TextColumn("Customer ID", width="medium"),
            "MonthlyCharges": st.column_config.NumberColumn("Monthly ($)", format="%.2f"),
            "TotalCharges": st.column_config.NumberColumn("Total ($)", format="%.2f"),
            "tenure": st.column_config.ProgressColumn("Tenure", min_value=0, max_value=72, format="%d mo"),
            "Churn": st.column_config.TextColumn("Churn"),
        },
    )

# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:40px 0 16px;border-top:1px solid rgba(99,102,241,0.1);margin-top:40px;">
    <span style="font-size:0.78rem;color:var(--text-muted);">
        Built with ❤️ using Streamlit &nbsp;·&nbsp; ChurnGuard Pro &nbsp;·&nbsp; Powered by Gradient Boosting
    </span>
</div>
""", unsafe_allow_html=True)
