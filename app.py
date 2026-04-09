"""
CausalML — Uplift Modeling Platform
4-step wizard: Data → Setup → Model → Results
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

# ── Service layer (streamlit_app/) ───────────────────────────────────────
from streamlit_app.constants import (
    LEARNER_OPTIONS,
    BASE_MODEL_OPTIONS,
    PREFERRED_COLUMN_NAMES,
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_STATE,
)
from streamlit_app.models import RunConfig, ValidationReport
from streamlit_app.services.data_io import load_demo_dataset, load_csv_from_bytes, file_to_hash
from streamlit_app.services.validation import validate_dataset_selection, guess_default_column
from streamlit_app.services.training import train_and_compare

# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CausalML",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600&family=Geist+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Geist', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
.stApp { background-color: #0a0a0f; color: #94a3b8; }

section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2433;
}
section[data-testid="stSidebar"] > div:first-child { padding: 20px 16px !important; }

.block-container { padding: 48px 32px 64px 32px !important; max-width: 1100px !important; }

/* ── Tabs ── */
div[data-baseweb="tab-list"] {
    background: transparent !important; border: none !important;
    border-bottom: 1px solid #1e2433 !important; gap: 0 !important; padding: 0 !important;
}
button[data-baseweb="tab"] {
    font-family: 'Geist', sans-serif !important; font-size: 13px !important;
    font-weight: 500 !important; color: #64748b !important;
    background: transparent !important; border: none !important;
    border-bottom: 2px solid transparent !important; padding: 10px 18px !important;
    border-radius: 0 !important; margin: 0 !important;
}
button[data-baseweb="tab"]:hover { color: #94a3b8 !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #f1f5f9 !important; border-bottom: 2px solid #6366f1 !important;
    background: transparent !important;
}
div[data-baseweb="tab-panel"] { padding: 20px 0 0 0 !important; }

/* ── Buttons ── */
.stButton > button {
    background: #6366f1 !important; color: #fff !important; border: none !important;
    border-radius: 6px !important; font-family: 'Geist', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important;
    padding: 8px 20px !important; width: 100% !important; box-shadow: none !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #4f52d9 !important; }
.stDownloadButton > button {
    background: #13131a !important; color: #6366f1 !important;
    border: 1px solid #1e2433 !important; border-radius: 6px !important;
    font-family: 'Geist', sans-serif !important; font-size: 12px !important;
    font-weight: 500 !important; width: 100% !important;
}

/* ── Selectbox / multiselect ── */
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background-color: #13131a !important; border: 1px solid #1e2433 !important;
    border-radius: 6px !important; font-size: 12px !important;
    color: #94a3b8 !important; font-family: 'Geist', sans-serif !important;
}

/* ── Radio ── */
.stRadio > label { font-size: 12px !important; color: #64748b !important; }
.stRadio div[role="radiogroup"] label { font-size: 13px !important; color: #94a3b8 !important; }

/* ── Slider ── */
.stSlider label { font-size: 12px !important; color: #64748b !important; }

/* ── Checkbox ── */
.stCheckbox label { font-size: 13px !important; color: #94a3b8 !important; }

/* ── Tables / dataframes ── */
.stDataFrame { border: 1px solid #1e2433 !important; border-radius: 6px !important; overflow: hidden !important; }
.stDataFrame thead tr th { background: #13131a !important; }

/* ── Remove chrome ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none !important; }
header[data-testid="stHeader"] { background: #0a0a0f !important; border-bottom: 1px solid #1e2433; }

/* ── Step indicator (sidebar) ── */
.step-item {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 10px; border-radius: 6px; margin: 2px 0;
    font-family: 'Geist', sans-serif; font-size: 13px;
}
.step-item.done  { color: #10b981; background: rgba(16,185,129,0.07); }
.step-item.curr  { color: #f1f5f9; background: #13131a; font-weight: 600; }
.step-item.lock  { color: #334155; }
.step-dot        { width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 600; flex-shrink: 0; }
.step-dot.done   { background: #10b981; color: #041313; }
.step-dot.curr   { background: #6366f1; color: #fff; }
.step-dot.lock   { background: #1e2433; color: #475569; }

/* ── Page header ── */
.page-hdr         { padding: 4px 0 20px 0; border-bottom: 1px solid #1e2433; margin-bottom: 24px; }
.page-hdr-title   { font-size: 18px; font-weight: 600; color: #f1f5f9; letter-spacing: -0.02em; font-family: 'Geist', sans-serif; }
.page-hdr-desc    { font-size: 13px; color: #64748b; margin-top: 4px; font-family: 'Geist', sans-serif; }

/* ── Section header ── */
.sec-hdr {
    font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
    color: #64748b; text-transform: uppercase; font-family: 'Geist', sans-serif;
    padding-top: 28px; padding-bottom: 6px;
    border-bottom: 1px solid #1e2433; margin-bottom: 14px; display: block;
}

/* ── Card ── */
.card {
    background: #13131a; border: 1px solid #1e2433; border-radius: 6px;
    padding: 16px 20px; margin-bottom: 12px;
}
.card-lbl { font-size: 11px; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: #64748b; font-family: 'Geist', sans-serif; margin-bottom: 4px; }
.card-val { font-size: 22px; font-weight: 600; color: #f1f5f9; letter-spacing: -0.02em; font-family: 'Geist', sans-serif; }
.card-sub { font-size: 11px; color: #475569; margin-top: 3px; font-family: 'Geist', sans-serif; line-height: 1.5; }

/* ── Inline badge ── */
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; font-family: 'Geist Mono', monospace; }
.badge-g { background: rgba(16,185,129,0.12); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-a { background: rgba(245,158,11,0.12); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-r { background: rgba(239,68,68,0.12);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }

/* ── Notif strip ── */
.n-ok   { background: rgba(16,185,129,0.08); border-left: 2px solid #10b981; color: #6ee7b7;  font-size: 12px; padding: 8px 14px; border-radius: 0 4px 4px 0; font-family: 'Geist', sans-serif; margin: 12px 0; }
.n-err  { background: rgba(239,68,68,0.08);  border-left: 2px solid #ef4444; color: #fca5a5;  font-size: 12px; padding: 8px 14px; border-radius: 0 4px 4px 0; font-family: 'Geist', sans-serif; margin: 12px 0; }
.n-warn { background: rgba(245,158,11,0.08); border-left: 2px solid #f59e0b; color: #fcd34d;  font-size: 12px; padding: 8px 14px; border-radius: 0 4px 4px 0; font-family: 'Geist', sans-serif; margin: 12px 0; }
.n-info { background: rgba(99,102,241,0.08); border-left: 2px solid #6366f1; color: #a5b4fc;  font-size: 12px; padding: 8px 14px; border-radius: 0 4px 4px 0; font-family: 'Geist', sans-serif; margin: 12px 0; }

/* ── Info box ── */
.ib { border-left: 2px solid #6366f1; background: rgba(99,102,241,0.06); padding: 10px 14px; border-radius: 0 4px 4px 0; font-size: 12px; color: #94a3b8; margin: 10px 0; line-height: 1.7; font-family: 'Geist', sans-serif; }
.ib code { font-family: 'Geist Mono', monospace; font-size: 11px; }

/* ── Sidebar brand ── */
.sb-logo { font-size: 15px; font-weight: 600; color: #f1f5f9; font-family: 'Geist', sans-serif; letter-spacing: -0.01em; }
.sb-sub  { font-size: 11px; color: #475569; font-family: 'Geist Mono', monospace; margin-top: 2px; }
.sb-div  { height: 1px; background: #1e2433; margin: 14px 0; }
.sb-lbl  { font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #475569; font-family: 'Geist', sans-serif; margin: 16px 0 8px 0; display: block; }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE ─────────────────────────────────────────────────────────
_DEFAULTS = {
    "step":             1,
    "raw_df":           None,
    "dataset_source":   None,
    "col_selection":    None,   # dict with treatment/outcome/features/true_cate/propensity
    "val_report":       None,   # ValidationReport
    "run_config":       None,   # RunConfig
    "train_result":     None,   # TrainingResult
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_from(step: int):
    """Clear state for step and everything after it."""
    if step <= 1:
        st.session_state.raw_df         = None
        st.session_state.dataset_source = None
    if step <= 2:
        st.session_state.col_selection  = None
        st.session_state.val_report     = None
    if step <= 3:
        st.session_state.run_config     = None
        st.session_state.train_result   = None
    if step <= st.session_state.step:
        st.session_state.step = max(1, step)


def go_to(step: int):
    st.session_state.step = step


# ── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">CausalML</div>
    <div class="sb-sub">Uplift Modeling Platform</div>
    <div class="sb-div"></div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sb-lbl">Progress</span>', unsafe_allow_html=True)

    steps_meta = [
        (1, "Data",    "Load dataset"),
        (2, "Setup",   "Map columns"),
        (3, "Model",   "Train learners"),
        (4, "Results", "View output"),
    ]
    cur = st.session_state.step
    # Determine which steps are unlocked
    unlocked = {1}
    if st.session_state.raw_df is not None:          unlocked.add(2)
    if st.session_state.val_report is not None and st.session_state.val_report.is_valid: unlocked.add(3)
    if st.session_state.train_result is not None:    unlocked.add(4)

    for num, name, desc in steps_meta:
        if num < cur or (num in unlocked and num != cur):
            cls, dot_cls, dot_icon = "done", "done", "✓"
        elif num == cur:
            cls, dot_cls, dot_icon = "curr", "curr", str(num)
        else:
            cls, dot_cls, dot_icon = "lock", "lock", str(num)

        st.markdown(
            f'<div class="step-item {cls}">'
            f'<div class="step-dot {dot_cls}">{dot_icon}</div>'
            f'<div><div>{name}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Quick-jump buttons for completed steps
    if len(unlocked) > 1:
        st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)
        st.markdown('<span class="sb-lbl">Jump to</span>', unsafe_allow_html=True)
        for num, name, _ in steps_meta:
            if num in unlocked and num != cur:
                if st.button(name, key=f"jump_{num}"):
                    go_to(num)
                    st.rerun()

    if st.session_state.train_result is not None:
        st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)
        if st.button("Start over", key="restart"):
            for k, v in _DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()


# ── HELPERS ───────────────────────────────────────────────────────────────
def page_header(title: str, desc: str = ""):
    st.markdown(
        f'<div class="page-hdr"><div class="page-hdr-title">{title}</div>'
        + (f'<div class="page-hdr-desc">{desc}</div>' if desc else "")
        + '</div>',
        unsafe_allow_html=True,
    )

def sec(label: str):
    st.markdown(f'<span class="sec-hdr">{label}</span>', unsafe_allow_html=True)

def notif(msg: str, kind: str = "info"):
    st.markdown(f'<div class="n-{kind}">{msg}</div>', unsafe_allow_html=True)

def ib(msg: str):
    st.markdown(f'<div class="ib">{msg}</div>', unsafe_allow_html=True)

def badge(text: str, color: str = "g") -> str:
    return f'<span class="badge badge-{color}">{text}</span>'

def card_metric(label: str, value: str, sub: str = ""):
    st.markdown(
        f'<div class="card"><div class="card-lbl">{label}</div>'
        f'<div class="card-val">{value}</div>'
        + (f'<div class="card-sub">{sub}</div>' if sub else "")
        + "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1 — DATA
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.step == 1:
    page_header(
        "Step 1 — Load Data",
        "Use the built-in synthetic dataset or upload your own CSV file.",
    )

    tab_demo, tab_upload = st.tabs(["Sample data", "Upload CSV"])

    with tab_demo:
        st.markdown("")
        st.markdown(
            '<p style="font-size:13px;color:#94a3b8;font-family:Geist,sans-serif;">'
            "The built-in dataset uses "
            "<code style='font-family:Geist Mono,monospace;font-size:11px;'>causalml.dataset.synthetic_data(mode=2, sigma=0.5)</code> — "
            "a randomised trial (constant propensity=0.5) with a learnable, nonlinear CATE function "
            "and known ground-truth <code style='font-family:Geist Mono,monospace;font-size:11px;'>true_cate</code>. "
            "Expect normalized AUUC <strong>~0.88–0.98</strong> with a good learner."
            "</p>",
            unsafe_allow_html=True,
        )

        n_rows = st.selectbox(
            "Sample size",
            [500, 1000, 2000, 5000],
            index=2,  # default 2000
            help="Larger = more accurate CATE estimates, slower training",
        )

        if st.button("Load sample data", key="load_demo"):
            with st.spinner("Generating synthetic dataset…"):
                df = load_demo_dataset(n_rows=n_rows, random_state=42)
            reset_from(2)
            st.session_state.raw_df         = df
            st.session_state.dataset_source = f"Synthetic demo · {n_rows:,} rows"
            go_to(2)
            st.rerun()

        ib(
            "<strong>About this dataset</strong> — Setup B (Nie &amp; Wager 2018): "
            "8 numeric features, binary treatment, continuous outcome, "
            "ground-truth ITE (<code>true_cate = X1 + log(1+exp(X2))</code>), and propensity score. "
            "Use <strong>X-Learner + Random Forest</strong> for the best normalized AUUC."
        )

    with tab_upload:
        st.markdown("")
        uploaded = st.file_uploader(
            "CSV file",
            type="csv",
            label_visibility="collapsed",
            help="Must contain at least a treatment column (0/1) and a numeric outcome column.",
        )
        if uploaded is not None:
            file_bytes = uploaded.getvalue()
            df = load_csv_from_bytes(file_bytes, uploaded.name)
            if st.button("Use this file", key="use_upload"):
                reset_from(2)
                st.session_state.raw_df         = df
                st.session_state.dataset_source = f"Uploaded · {uploaded.name}"
                go_to(2)
                st.rerun()

            sec("Preview")
            st.dataframe(df.head(8), width="stretch")

        st.markdown("")
        ib(
            "<strong>Required columns:</strong> one binary treatment column (0/1) and one numeric outcome column. "
            "All other numeric and low-cardinality categorical columns are auto-detected as features.<br>"
            "Optional: <code>true_cate</code> (ground-truth ITE) and <code>propensity</code> — "
            "if present, the app computes additional accuracy metrics."
        )


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2 — SETUP
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    df: pd.DataFrame = st.session_state.raw_df
    page_header(
        "Step 2 — Column Setup",
        f"Map your columns · {st.session_state.dataset_source}",
    )

    all_cols = list(df.columns)
    notif(f"{len(df):,} rows · {len(df.columns)} columns", "ok")

    sec("Column Mapping")
    ib("The app auto-detected likely column roles. Review and adjust if needed.")

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        treatment_col = st.selectbox(
            "Treatment column (0 = control, 1 = treated)",
            all_cols,
            index=all_cols.index(guess_default_column(all_cols, "treatment") or all_cols[0]),
        )
        outcome_col = st.selectbox(
            "Outcome column (numeric)",
            all_cols,
            index=all_cols.index(guess_default_column(all_cols, "outcome") or all_cols[1]),
        )

    with col_right:
        cate_guess = guess_default_column(all_cols, "true_cate")
        prop_guess = guess_default_column(all_cols, "propensity")

        true_cate_col = st.selectbox(
            "True CATE column (optional — enables accuracy metrics)",
            ["— none —"] + all_cols,
            index=(["— none —"] + all_cols).index(cate_guess) if cate_guess else 0,
        )
        true_cate_col = None if true_cate_col == "— none —" else true_cate_col

        propensity_col = st.selectbox(
            "Propensity score column (optional)",
            ["— none —"] + all_cols,
            index=(["— none —"] + all_cols).index(prop_guess) if prop_guess else 0,
        )
        propensity_col = None if propensity_col == "— none —" else propensity_col

    # Feature multiselect — exclude reserved cols
    reserved = {treatment_col, outcome_col}
    if true_cate_col: reserved.add(true_cate_col)
    if propensity_col: reserved.add(propensity_col)
    candidate_features = [c for c in all_cols if c not in reserved]

    sec("Feature Columns")
    selected_features = st.multiselect(
        "Select features to use for training",
        candidate_features,
        default=candidate_features,
        label_visibility="collapsed",
    )

    if st.button("Validate and continue →", key="validate"):
        if not selected_features:
            notif("Select at least one feature column.", "err")
        else:
            report = validate_dataset_selection(
                df=df,
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                true_cate_col=true_cate_col,
                propensity_col=propensity_col,
                exclude_cols=[c for c in all_cols if c not in selected_features
                              and c not in reserved],
            )

            if report.errors:
                for err in report.errors:
                    notif(err, "err")
            else:
                st.session_state.val_report = report
                st.session_state.col_selection = {
                    "treatment":   treatment_col,
                    "outcome":     outcome_col,
                    "features":    selected_features,
                    "true_cate":   true_cate_col,
                    "propensity":  propensity_col,
                }
                reset_from(3)
                go_to(3)
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3 — MODEL
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    df   = st.session_state.raw_df
    cols = st.session_state.col_selection

    page_header(
        "Step 3 — Choose Model",
        "Select which meta-learners to train and how to split the data.",
    )

    # Learner explanations
    sec("Meta-Learner Algorithms")

    learner_desc = {
        "S": ("S-Learner",
              "Trains <em>one</em> model with treatment T as an extra feature. CATE = prediction(T=1) − prediction(T=0). "
              "Simple, fast. May underweight treatment signal on large datasets."),
        "T": ("T-Learner",
              "Trains <em>two</em> separate models — one on treated, one on control. CATE = μ₁(X) − μ₀(X). "
              "Better signal isolation; higher variance with small groups."),
        "X": ("X-Learner",
              "Cross-imputes counterfactual effects then combines them with propensity weighting. "
              "Best for imbalanced treatment ratios."),
        "R": ("R-Learner",
              "Residual decomposition: regresses (Y − m̂(X)) on (T − ê(X)) to estimate CATE. "
              "Doubly-robust; often highest accuracy when propensity estimation is good."),
    }

    cols_l = st.columns(4, gap="medium")
    for i, (code, (label, desc)) in enumerate(learner_desc.items()):
        with cols_l[i]:
            st.markdown(
                f'<div class="card" style="min-height:130px;">'
                f'<div style="font-size:13px;font-weight:600;color:#f1f5f9;font-family:Geist,sans-serif;margin-bottom:8px;">{label}</div>'
                f'<div style="font-size:12px;color:#64748b;font-family:Geist,sans-serif;line-height:1.6;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    sec("Configuration")
    cfg_a, cfg_b = st.columns(2, gap="large")

    with cfg_a:
        st.markdown('<span style="font-size:12px;color:#64748b;font-family:Geist,sans-serif;">Which learners to run</span>', unsafe_allow_html=True)
        selected_learners = []
        for code, label in LEARNER_OPTIONS.items():
            checked = st.checkbox(label, value=(code == "T"), key=f"chk_{code}")
            if checked:
                selected_learners.append(code)

        base_model = st.selectbox("Base estimator", list(BASE_MODEL_OPTIONS.keys()),
                                  format_func=lambda k: BASE_MODEL_OPTIONS[k])

    with cfg_b:
        test_pct = st.slider("Test split", 10, 40, 20, 5, format="%d%%",
                              help="Fraction of data held out for evaluation")
        test_size = test_pct / 100.0
        st.markdown(
            f'<p style="font-size:12px;color:#64748b;font-family:Geist Mono,monospace;margin-top:4px;">'
            f'Train: {len(df) - int(len(df)*test_size):,} rows &nbsp;·&nbsp; '
            f'Test: {int(len(df)*test_size):,} rows</p>',
            unsafe_allow_html=True,
        )

        random_state = st.number_input("Random seed", value=DEFAULT_RANDOM_STATE,
                                        min_value=0, step=1)

    if not selected_learners:
        notif("Select at least one learner.", "warn")

    elif st.button("Run analysis →", key="run"):
        run_config = RunConfig(
            treatment_col=cols["treatment"],
            outcome_col=cols["outcome"],
            feature_cols=cols["features"],
            learner_names=selected_learners,
            base_model_name=base_model,
            true_cate_col=cols["true_cate"],
            propensity_col=cols["propensity"],
            test_size=test_size,
            random_state=int(random_state),
        )
        st.session_state.run_config = run_config

        progress_bar = st.progress(0, text="Starting…")
        progress_text = st.empty()

        def _progress(done, total, msg):
            frac = done / total if total else 0
            progress_bar.progress(frac, text=msg)
            progress_text.markdown(
                f'<span style="font-size:12px;color:#64748b;font-family:Geist,sans-serif;">{msg}</span>',
                unsafe_allow_html=True,
            )

        with st.spinner(""):
            result = train_and_compare(
                df=df,
                validation_report=st.session_state.val_report,
                config=run_config,
                progress_callback=_progress,
            )

        progress_bar.progress(1.0, text="Done.")
        progress_text.empty()
        st.session_state.train_result = result
        go_to(4)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4 — RESULTS
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    result  = st.session_state.train_result
    config  = result.config
    outputs = result.learner_outputs   # dict[str, LearnerArtifacts]

    page_header(
        "Step 4 — Results",
        f"{' · '.join(LEARNER_OPTIONS[k] for k in config.learner_names)}  ·  "
        f"{BASE_MODEL_OPTIONS[config.base_model_name]}  ·  "
        f"test split {int(config.test_size*100)}%",
    )

    # ── A. ATE SUMMARY ────────────────────────────────────────────────────
    sec("A  —  Average Treatment Effect")
    ib(
        "<strong>ATE (Average Treatment Effect)</strong> is the mean predicted uplift across all customers. "
        "It tells you: on average, how many additional outcome units does treatment cause? "
        "A positive ATE means treatment helps on average; a negative ATE means it hurts."
    )

    ate_rows = []
    for code, art in outputs.items():
        m = art.metrics
        ate_rows.append({
            "Learner":              art.learner_label,
            "Predicted ATE (test)": f"{m['predicted_ate_test']:+.4f}",
            "Avg CATE (full data)": f"{m['predicted_ate_full']:+.4f}",
            "Fit time":             f"{m['fit_seconds']:.1f}s",
        })
    st.dataframe(
        pd.DataFrame(ate_rows).set_index("Learner"),
        width="stretch",
    )

    # ── B. MODEL QUALITY (only if true_cate available) ────────────────────
    has_true = config.true_cate_col is not None
    if has_true:
        sec("B  —  Model Quality")
        ib(
            "<strong>Normalized AUUC</strong> measures how well the model ranks customers by true uplift. "
            "1.0 = perfect ranking, 0.5 = random, &lt;0.5 = worse than random. "
            "<strong>CATE RMSE</strong> is the average prediction error in the same units as the outcome. "
            "<strong>Gain @ top 25%</strong> is the total true uplift captured by targeting the top quarter of predicted scores."
        )

        quality_rows = []
        for code, art in outputs.items():
            m = art.metrics
            n_auuc = m.get("normalized_auuc", float("nan"))
            rmse   = m.get("cate_rmse",       float("nan"))
            g25    = m.get("gain_at_top_25pct", float("nan"))

            if n_auuc >= 0.7:   auuc_badge = badge("Good", "g")
            elif n_auuc >= 0.5: auuc_badge = badge("OK", "a")
            else:               auuc_badge = badge("Weak", "r")

            quality_rows.append({
                "Learner":            art.learner_label,
                "Normalized AUUC":    f"{n_auuc:.4f}",
                "AUUC Interpretation":auuc_badge,
                "CATE RMSE":          f"{rmse:.4f}" if not np.isnan(rmse) else "—",
                "Gain @ top 25%":     f"{g25:.4f}"  if not np.isnan(g25)  else "—",
            })

        st.write(
            pd.DataFrame(quality_rows).set_index("Learner").to_html(escape=False),
            unsafe_allow_html=True,
        )
        st.markdown("")

    # ── C. CATE DECILE TABLE ──────────────────────────────────────────────
    sec("C  —  CATE Deciles")
    ib(
        "Customers in the test set sorted by predicted CATE into 10 equal groups. "
        "D10 = top 10% (highest predicted uplift), D1 = bottom 10%. "
        "A well-calibrated model shows a clear monotone increase from D1 to D10."
    )

    learner_tabs = st.tabs([LEARNER_OPTIONS[k] for k in outputs])
    for tab_obj, (code, art) in zip(learner_tabs, outputs.items()):
        with tab_obj:
            cate_vals = art.test_cate.values
            try:
                labels  = [f"D{i}" for i in range(1, 11)]
                deciles = pd.qcut(cate_vals, q=10, labels=labels, duplicates="drop")
                dec_df  = (
                    pd.DataFrame({"Decile": deciles, "Predicted CATE": cate_vals})
                    .groupby("Decile", observed=True)
                    .agg(
                        Avg_CATE=("Predicted CATE", "mean"),
                        Min_CATE=("Predicted CATE", "min"),
                        Max_CATE=("Predicted CATE", "max"),
                        Count=("Predicted CATE", "count"),
                    )
                    .reset_index()
                    .sort_values("Decile", ascending=False)
                )
                dec_df["Avg CATE"] = dec_df["Avg_CATE"].map("{:+.4f}".format)
                dec_df["Min CATE"] = dec_df["Min_CATE"].map("{:+.4f}".format)
                dec_df["Max CATE"] = dec_df["Max_CATE"].map("{:+.4f}".format)
                dec_df["% of test"] = (dec_df["Count"] / len(cate_vals) * 100).map("{:.1f}%".format)
                st.dataframe(
                    dec_df[["Decile", "Count", "% of test", "Avg CATE", "Min CATE", "Max CATE"]].set_index("Decile"),
                    width="stretch",
                )
            except Exception as exc:
                notif(f"Could not build decile table: {exc}", "warn")

    # ── D. TARGETING EFFICIENCY ───────────────────────────────────────────
    sec("D  —  Targeting Efficiency")
    ib(
        "If you rank all customers by predicted CATE (highest first) and target the top X%, "
        "what share of the <em>total model-predicted uplift</em> do you capture? "
        "A good model captures most uplift by targeting a small fraction."
    )

    tgt_tabs = st.tabs([LEARNER_OPTIONS[k] for k in outputs])
    for tab_obj, (code, art) in zip(tgt_tabs, outputs.items()):
        with tab_obj:
            cate_vals = art.test_cate.values
            total_gain = art.gain_curve["gain"].iloc[-1]
            tgt_rows = []
            for pct in [10, 20, 30, 40, 50]:
                n_top  = max(1, int(len(cate_vals) * pct / 100))
                top_ix = np.argsort(cate_vals)[::-1][:n_top]
                if has_true:
                    true_arr = result.prepared_data.true_cate_test.values
                    captured = float(true_arr[top_ix].sum())
                    total_t  = float(true_arr.sum()) if float(true_arr.sum()) != 0 else float("nan")
                    pct_capt = f"{captured/total_t*100:.1f}%" if not np.isnan(total_t) else "—"
                    avg_true = f"{true_arr[top_ix].mean():+.4f}"
                else:
                    row_pct  = art.gain_curve["population_fraction"]
                    # interpolate gain at this percentile
                    idx_near = (row_pct - pct/100).abs().idxmin()
                    captured = art.gain_curve["gain"].iloc[idx_near]
                    pct_capt = f"{captured/total_gain*100:.1f}%" if total_gain else "—"
                    avg_true = "—"

                tgt_rows.append({
                    "Target top %":            f"{pct}%",
                    "Customers targeted":      f"{n_top:,}",
                    "% uplift captured":       pct_capt,
                    "Avg predicted CATE":      f"{cate_vals[top_ix].mean():+.4f}",
                    "Avg true CATE (if known)":avg_true,
                })
            st.dataframe(pd.DataFrame(tgt_rows).set_index("Target top %"), width="stretch")

    # ── E. UPLIFT SEGMENTS ────────────────────────────────────────────────
    sec("E  —  Uplift Segments")
    ib(
        "Customers split into three groups by predicted CATE: "
        "<strong>Top 25%</strong> (Persuadables — target these), "
        "<strong>Middle 50%</strong> (Uncertain), "
        "<strong>Bottom 25%</strong> (Sleeping Dogs or Lost Causes — avoid targeting)."
    )

    seg_tabs = st.tabs([LEARNER_OPTIONS[k] for k in outputs])
    for tab_obj, (code, art) in zip(seg_tabs, outputs.items()):
        with tab_obj:
            seg_df = art.segment_summary.copy()
            seg_df = seg_df.rename(columns={
                "segment":                "Segment",
                "user_count":             "Count",
                "average_predicted_cate": "Avg Predicted CATE",
                "average_outcome":        "Avg Outcome",
                "average_true_cate":      "Avg True CATE",
            })
            # Format numerics
            for col in ["Avg Predicted CATE", "Avg Outcome", "Avg True CATE"]:
                if col in seg_df.columns:
                    seg_df[col] = pd.to_numeric(seg_df[col], errors="coerce").map(
                        lambda v: f"{v:+.4f}" if not np.isnan(v) else "—"
                    )
            st.dataframe(seg_df.set_index("Segment") if "Segment" in seg_df.columns else seg_df,
                         width="stretch")

    # ── F. FEATURE IMPORTANCE ─────────────────────────────────────────────
    sec("F  —  Feature Importance")
    ib(
        "Importance is measured by fitting a surrogate Random Forest to predict CATE from features. "
        "Higher importance = that feature has more influence on which customers are classified as persuadable. "
        "This is a global model-level explanation, not a per-customer score."
    )

    fi_tabs = st.tabs([LEARNER_OPTIONS[k] for k in outputs])
    for tab_obj, (code, art) in zip(fi_tabs, outputs.items()):
        with tab_obj:
            fi_df = art.feature_importance.copy().head(8)
            fi_df["Rank"] = range(1, len(fi_df) + 1)
            fi_df["Importance"] = fi_df["importance"].map("{:.4f}".format)
            fi_df["Share"] = (fi_df["importance"] / fi_df["importance"].sum() * 100).map("{:.1f}%".format)
            st.dataframe(
                fi_df[["Rank", "original_feature", "Importance", "Share"]]
                .rename(columns={"original_feature": "Feature"})
                .set_index("Rank"),
                width="stretch",
            )

    # ── G. EXPORT ──────────────────────────────────────────────────────────
    sec("G  —  Export")
    export_df = result.predictions_full.copy()
    st.download_button(
        label=f"Download full predictions CSV  ({len(export_df):,} rows)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="causalml_predictions.csv",
        mime="text/csv",
    )
    ib(
        "The CSV contains the original columns plus one <code>{learner}_cate</code> column "
        "per trained learner — the individual-level predicted treatment effect for each row."
    )

    # Quick data snapshot at the bottom
    sec("Run Summary")
    st.dataframe(
        result.comparison_table[[
            "learner_label", "average_predicted_cate", "predicted_ate_test",
            "auuc_like", "normalized_auuc", "cate_rmse", "fit_seconds",
        ]].rename(columns={
            "learner_label":           "Learner",
            "average_predicted_cate":  "Avg CATE (full)",
            "predicted_ate_test":      "ATE (test)",
            "auuc_like":               "AUUC (raw)",
            "normalized_auuc":         "AUUC (norm.)",
            "cate_rmse":               "CATE RMSE",
            "fit_seconds":             "Fit time (s)",
        }).set_index("Learner"),
        width="stretch",
    )
