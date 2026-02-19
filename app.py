"""
Credit Risk Predictor — Premium Streamlit Application
===============================================================
Run from the Application/ directory:
    streamlit run app.py
"""

import os, sys, json, pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()
NB_DIR       = PROJECT_ROOT / "Python Notebooks"
MLRUNS       = PROJECT_ROOT / "mlruns"
sys.path.insert(0, str(NB_DIR))

# ── Environment ──────────────────────────────────────────────────────────────
_env_loaded = load_dotenv()
_tmp_key = os.getenv("OPENAI_API_KEY")
client_ai = OpenAI(api_key=_tmp_key) if _tmp_key else None

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Credit Risk Predictor",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# PREMIUM DARK CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: #0a0e1a;
    color: #e2e8f0;
}

/* Hide sidebar toggle */
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"] { display: none; }

/* Main container */
.block-container { max-width: 1100px; padding: 2rem 2rem 4rem 2rem; }

/* ── Typography ─────────────────────────────────────────────────────────── */
h1 { font-size: 2rem !important; font-weight: 700 !important;
     background: linear-gradient(135deg, #667eea, #764ba2);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     margin-bottom: 0 !important; }
h3 { color: #94a3b8 !important; font-weight: 500 !important;
     font-size: 0.85rem !important; text-transform: uppercase;
     letter-spacing: 1.2px; margin-top: 1.8rem !important;
     margin-bottom: 0.5rem !important; }

/* ── Input widgets ──────────────────────────────────────────────────────── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
div[data-baseweb="select"] > div {
    background: #131929 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    transition: border-color 0.2s;
}
div[data-baseweb="select"] > div:hover,
[data-testid="stNumberInput"] input:focus {
    border-color: #667eea !important;
}
[data-testid="stSlider"] { padding: 0; }
.stSlider [data-baseweb="slider"] { background: #131929; }

/* ── Section card ───────────────────────────────────────────────────────── */
.section-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.2rem;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 1rem !important; padding: 0.7rem 2.5rem !important;
    letter-spacing: 0.3px;
    transition: opacity 0.2s, transform 0.1s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

/* ── Result cards ───────────────────────────────────────────────────────── */
.result-card {
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.2rem;
    border: 1px solid;
}
.result-approve {
    background: linear-gradient(135deg, #0f2417, #0d1f12);
    border-color: #22c55e44;
}
.result-review {
    background: linear-gradient(135deg, #1c1a0f, #1a1800);
    border-color: #f59e0b44;
}
.result-decline {
    background: linear-gradient(135deg, #1f0f0f, #1a0a0a);
    border-color: #ef444444;
}
.verdict-text {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.3px;
}
.verdict-approve { color: #4ade80; }
.verdict-review  { color: #fbbf24; }
.verdict-decline { color: #f87171; }

/* ── Report box ─────────────────────────────────────────────────────────── */
.report-box {
    background: #131929;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    font-size: 0.96rem;
    line-height: 1.8;
    color: #cbd5e1;
}
.report-box b { color: #e2e8f0; }

/* ── Divider ────────────────────────────────────────────────────────────── */
.thin-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2d3748, transparent);
    margin: 2rem 0;
}

/* ── Header bar ─────────────────────────────────────────────────────────── */
.header-bar {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 2rem;
}
.subtitle { color: #64748b; font-size: 0.9rem; margin-top: 0.3rem; }

/* ── Make it up badge ───────────────────────────────────────────────────── */
.make-up-row { display: flex; gap: 0.6rem; margin-bottom: 1.4rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model …")
def load_model():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()
    model_name = "Credit Risk Model Final" # Updated to match v6 notebook logic
    versions = client.get_latest_versions(model_name)
    if versions:
        uri = f"models:/{model_name}/{versions[0].version}"
    else:
        uri = f"models:/{model_name}/1"
    return mlflow.sklearn.load_model(uri)

@st.cache_data
def load_categories():
    p = NB_DIR / "categories.json"
    return json.load(open(p)) if p.exists() else {}

def unwrap_xgboost(m):
    """Unwrap calibrated / pipeline to get raw XGBClassifier."""
    if hasattr(m, "calibrated_classifiers_"):
        return unwrap_xgboost(m.calibrated_classifiers_[0].estimator)
    if hasattr(m, "named_steps"):
        return unwrap_xgboost(list(m.named_steps.values())[-1])
    if hasattr(m, "steps"):
        return unwrap_xgboost(m.steps[-1][1])
    return m

def get_preproc_pipeline(m):
    """Return a Pipeline of all transformer steps (skip samplers & classifiers)."""
    if hasattr(m, "calibrated_classifiers_"):
        return get_preproc_pipeline(m.calibrated_classifiers_[0].estimator)
    if hasattr(m, "steps"):
        from sklearn.pipeline import Pipeline as _P
        from sklearn.base import ClassifierMixin
        transformer_steps = [
            (name, step) for name, step in m.steps
            if hasattr(step, "transform") and not isinstance(step, ClassifierMixin)
        ]
        return _P(transformer_steps) if transformer_steps else None
    if hasattr(m, "named_steps"):
        return get_preproc_pipeline(type("_P", (), {"steps": list(m.named_steps.items())})())
    return None

CATS = load_categories()

SAMPLES = {
    "Low Risk": {
        "CODE_GENDER": "F", "FLAG_OWN_CAR": 1, "FLAG_OWN_REALTY": 1,
        "CNT_CHILDREN": 0, "DEBT_TO_INCOME": 1.2, "AVERAGE_EXTERNAL_RATING": 0.72,
        "N_DOCUMENTS_PROVIDED": 3, "ADDITIONAL_DOC_PROVIDED": True,
        "TOT_PREV_APP": 3, "APPROVED_RATIO": 0.8, "REFUSED_RATIO": 0.1,
        "CANCELLED_RATIO": 0.1, "UNUSED_RATIO": 0.0,
        "NAME_TYPE_SUITE": "Unaccompanied", "NAME_INCOME_TYPE": "Working",
        "NAME_FAMILY_STATUS": "Married", "NAME_HOUSING_TYPE": "House / apartment",
        "REGION_RATING_CLIENT": 1, "YEARS_BIRTH": 42.0, "YEARS_EMPLOYED": 8.5,
        "YEARS_REGISTRATION": 15.0, "YEARS_ID_PUBLISH": 5.0,
        "YEARS_LAST_PHONE_CHANGE": 3.0, "ORG_GROUP": "Business",
        "OCCUPATION_TYPE_GROUPED": "WhiteCollar/Admin", "EDUCATION_LEVEL": "Higher Academic",
    },
    "High Risk": {
        "CODE_GENDER": "M", "FLAG_OWN_CAR": 0, "FLAG_OWN_REALTY": 0,
        "CNT_CHILDREN": 3, "DEBT_TO_INCOME": 8.9, "AVERAGE_EXTERNAL_RATING": 0.15,
        "N_DOCUMENTS_PROVIDED": 1, "ADDITIONAL_DOC_PROVIDED": False,
        "TOT_PREV_APP": 5, "APPROVED_RATIO": 0.2, "REFUSED_RATIO": 0.6,
        "CANCELLED_RATIO": 0.2, "UNUSED_RATIO": 0.0,
        "NAME_TYPE_SUITE": "Unaccompanied", "NAME_INCOME_TYPE": "Working",
        "NAME_FAMILY_STATUS": "Single / not married", "NAME_HOUSING_TYPE": "Rented apartment",
        "REGION_RATING_CLIENT": 3, "YEARS_BIRTH": 28.0, "YEARS_EMPLOYED": 0.5,
        "YEARS_REGISTRATION": 2.0, "YEARS_ID_PUBLISH": 0.5,
        "YEARS_LAST_PHONE_CHANGE": 0.1, "ORG_GROUP": "Unknown/Other",
        "OCCUPATION_TYPE_GROUPED": "Labour/LowSkill", "EDUCATION_LEVEL": "Medium Education",
    },
}

def random_sample() -> dict:
    """Generate a random but valid applicant profile using categories from CATS."""
    import random
    return {
        "CODE_GENDER":             random.choice(CATS.get("CODE_GENDER", ["M", "F"])),
        "FLAG_OWN_CAR":            random.choice([0, 1]),
        "FLAG_OWN_REALTY":         random.choice([0, 1]),
        "CNT_CHILDREN":            random.randint(0, 5),
        "DEBT_TO_INCOME":          round(random.uniform(0.5, 12.0), 2),
        "AVERAGE_EXTERNAL_RATING": round(random.uniform(0.05, 0.95), 2),
        "N_DOCUMENTS_PROVIDED":    random.randint(0, 5),
        "ADDITIONAL_DOC_PROVIDED": random.choice([True, False]),
        "TOT_PREV_APP":            random.randint(0, 10),
        "APPROVED_RATIO":          round(random.uniform(0.0, 1.0), 2),
        "REFUSED_RATIO":           round(random.uniform(0.0, 1.0), 2),
        "CANCELLED_RATIO":         round(random.uniform(0.0, 0.5), 2),
        "UNUSED_RATIO":            round(random.uniform(0.0, 0.3), 2),
        "NAME_TYPE_SUITE":         random.choice(CATS.get("NAME_TYPE_SUITE", ["Unaccompanied"])),
        "NAME_INCOME_TYPE":        random.choice(CATS.get("NAME_INCOME_TYPE", ["Working"])),
        "NAME_FAMILY_STATUS":      random.choice(CATS.get("NAME_FAMILY_STATUS", ["Married"])),
        "NAME_HOUSING_TYPE":       random.choice(CATS.get("NAME_HOUSING_TYPE", ["House / apartment"])),
        "REGION_RATING_CLIENT":    random.choice([1, 2, 3]),
        "YEARS_BIRTH":             round(random.uniform(20.0, 68.0), 1),
        "YEARS_EMPLOYED":          round(random.uniform(0.0, 30.0), 1),
        "YEARS_REGISTRATION":      round(random.uniform(0.0, 40.0), 1),
        "YEARS_ID_PUBLISH":        round(random.uniform(0.0, 20.0), 1),
        "YEARS_LAST_PHONE_CHANGE": round(random.uniform(0.0, 10.0), 1),
        "ORG_GROUP":               random.choice(CATS.get("ORG_GROUP", ["Business"])),
        "OCCUPATION_TYPE_GROUPED": random.choice(CATS.get("OCCUPATION_TYPE_GROUPED", ["WhiteCollar/Admin"])),
        "EDUCATION_LEVEL":         random.choice(CATS.get("EDUCATION_LEVEL", ["Medium Education"])),
    }


def cat_opts(field):
    return CATS.get(field, [])

def sbox(label, field, val):
    opts = cat_opts(field)
    idx  = opts.index(val) if val in opts else 0
    return st.selectbox(label, opts, index=idx, key=f"w_{field}")

def numinp(label, val, mn=0.0, mx=None, step=0.01, fmt="%.2f"):
    kw = dict(label=label, value=float(val), min_value=float(mn), step=step, format=fmt)
    if mx: kw["max_value"] = float(mx)
    return st.number_input(**kw)

def yn_box(label, field, val):
    """Yes/No selectbox returning 0 or 1."""
    choice = st.selectbox(label, ["No", "Yes"],
                          index=1 if int(val) == 1 else 0,
                          key=f"w_{field}")
    return 1 if choice == "Yes" else 0


# ══════════════════════════════════════════════════════════════════════════════
# LLM-STYLE REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_report_llm(data: dict, prob: float, pred: int, shap_summary: str = "") -> str:
    """Generate a dynamic, professional credit report using OpenAI GPT."""
    try:
        model_decision = "DECLINE" if prob >= 0.5 else ("REVIEW" if prob >= 0.25 else "APPROVE")
        
        prompt = f"""
We have a customer credit risk model that predicts the probability of default (PD).
Below is the data for a specific applicant and the key drivers of their risk score (SHAP values).

MODEL PREDICTION:
- Predicted Probability of Default (PD): {prob:.1%}
- Verdict: {model_decision}

APPLICANT DATA:
{json.dumps(data, indent=2)}

SHAP EXPLANATION (Feature Contributions):
{shap_summary}

Please write a short, structured **credit risk report** based *only* on the provided data. 
Do not assume facts (like "unemployment") unless the data explicitly supports it (e.g., YEARS_EMPLOYED is 0 or negative).

SECTIONS:
1. **Model Prediction Summary** – Summarise the predicted default probability and verdict.
2. **Key Drivers of Risk** – Explain the top SHAP features. Note: A POSITIVE SHAP value increases the probability of default (adds risk), while a NEGATIVE value decreases it (reduces risk).
3. **Recommendations for the Customer** – Actionable steps to improve their specific profile.
4. **Advice for the Banker** – Strategic advice and specific red flags to investigate.
5. **Next Steps** – Concrete actions for both parties.

Be concise, professional, and clear. Format the output with HTML tags like <b> and <br> for a Streamlit display.
Do NOT use markdown headers (#), use <b> for internal section titles.
"""
        response = client_ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a very experienced financial analyst and credit risk expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        content = response.choices[0].message.content
        return f'<div class="report-box">{content}</div>'

    except Exception as e:
        # Show the error in the main page (sidebar is CSS-hidden)
        st.warning(f"OpenAI API call failed — showing fallback report. Error: `{e}`")
        return generate_report_legacy(data, prob, pred)

def generate_report_legacy(data: dict, prob: float, pred: int) -> str:
    """Original rule-based report generator (fallback)."""
    verdict = "HIGH RISK" if prob >= 0.5 else ("MEDIUM RISK" if prob >= 0.25 else "LOW RISK")
    rec     = "decline" if prob >= 0.5 else ("review" if prob >= 0.25 else "approve")

    flags = []
    if data["DEBT_TO_INCOME"] > 5:
        flags.append(f"a high debt-to-income ratio of {data['DEBT_TO_INCOME']:.1f}")
    if data["AVERAGE_EXTERNAL_RATING"] < 0.3:
        flags.append(f"a low external credit rating of {data['AVERAGE_EXTERNAL_RATING']:.2f}")
    if data["REFUSED_RATIO"] > 0.4:
        flags.append(f"a high previous refusal rate of {data['REFUSED_RATIO']:.0%}")
    if data["YEARS_EMPLOYED"] < 1:
        flags.append("very short employment history (under 1 year)")
    if data.get("REGION_RATING_CLIENT") == 3:
        flags.append("a high-risk region rating (3/3)")

    report = f"""
<div class="report-box">
<b>Executive Summary (Local Engine)</b><br>
This application has been assessed as <b>{verdict}</b> (PD: {prob:.1%}). 
The internal recommendation is to <b>{rec}</b> this loan request.
<br><br>
<b>Key Factors:</b><br>
{', '.join(flags) if flags else 'No major red flags identified by the local heuristic engine.'}
</div>
"""
    return report

# For backward compatibility / simple entry point
def generate_report(data: dict, prob: float, pred: int, shap_summary: str = "") -> str:
    if client_ai and client_ai.api_key:
        return generate_report_llm(data, prob, pred, shap_summary)
    return generate_report_legacy(data, prob, pred)


# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-bar">
  <div>
    <h1>Credit Risk Predictor</h1>
    <div class="subtitle">An AI-powered system for credit decisioning and model performance monitoring.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── MAIN TABS ────────────────────────────────────────────────────────────────
tab_assess, tab_monitor = st.tabs(["Risk Assessment", "Model Monitoring"])

with tab_assess:
    # Quick-fill button
    col_lbl, col_high, col_sp = st.columns([2, 1, 5])
    with col_lbl:
        st.markdown("<div style='padding-top:0.5rem;color:#64748b;font-size:0.85rem;'>Quick Fill:</div>",
                    unsafe_allow_html=True)
    with col_high:
        fill_random = st.button("Make It Up")

    # Initialize session state for applicant data if not present
    if "D" not in st.session_state:
        st.session_state["D"] = SAMPLES["Low Risk"].copy()

    # Determine defaults
    if fill_random:
        st.session_state["D"] = random_sample()
        st.rerun() # Force a rerun so all widgets see the new values immediately

    D = st.session_state["D"]


    # ════════════════════════════════════════════════════════════════════════════
    # INPUT FORM — all 26 fields on one page
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)

    # ── Section 1: Personal Details ──────────────────────────────────────────────
    st.markdown("### Personal Details")
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gender = sbox("Gender", "CODE_GENDER", D["CODE_GENDER"])
        with c2:
            own_car = yn_box("Owns a Car", "FLAG_OWN_CAR", D["FLAG_OWN_CAR"])
        with c3:
            own_realty = yn_box("Owns Real Estate", "FLAG_OWN_REALTY", D["FLAG_OWN_REALTY"])
        with c4:
            cnt_children = st.number_input("Number of Children", min_value=0, max_value=20,
                                           value=int(D["CNT_CHILDREN"]), step=1, key="w_children")

        c1, c2, c3 = st.columns(3)
        with c1:
            family_status = sbox("Family Status", "NAME_FAMILY_STATUS", D["NAME_FAMILY_STATUS"])
        with c2:
            type_suite = sbox("Accompanied By", "NAME_TYPE_SUITE", D["NAME_TYPE_SUITE"])
        with c3:
            add_doc = st.selectbox("Additional Docs Provided", ["No", "Yes"],
                                   index=1 if D["ADDITIONAL_DOC_PROVIDED"] else 0,
                                   key="w_adddoc")
            add_doc_val = add_doc == "Yes"

    # ── Section 2: Age & Life Events ─────────────────────────────────────────────
    st.markdown("### Age & Life History (years)")
    with st.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            years_birth = numinp("Age", D["YEARS_BIRTH"], mn=18, mx=100, step=0.1)
        with c2:
            years_employed = numinp("Years Employed", D["YEARS_EMPLOYED"], mn=-1, mx=60, step=0.1)
        with c3:
            years_registration = numinp("Years Registered", D["YEARS_REGISTRATION"], mn=0, mx=60, step=0.1)
        with c4:
            years_id_publish = numinp("Years Since ID", D["YEARS_ID_PUBLISH"], mn=0, mx=30, step=0.1)
        with c5:
            years_phone = numinp("Years Since Phone Change", D["YEARS_LAST_PHONE_CHANGE"], mn=0, mx=30, step=0.1)

    # ── Section 3: Financial & Employment ────────────────────────────────────────
    st.markdown("### Financial & Employment")
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            debt_to_income = numinp("Debt to Income Ratio", D["DEBT_TO_INCOME"], mn=0.0, mx=50.0)
        with c2:
            income_type = sbox("Income Type", "NAME_INCOME_TYPE", D["NAME_INCOME_TYPE"])
        with c3:
            education = sbox("Education Level", "EDUCATION_LEVEL", D["EDUCATION_LEVEL"])
        with c4:
            housing = sbox("Housing Type", "NAME_HOUSING_TYPE", D["NAME_HOUSING_TYPE"])

        c1, c2, c3 = st.columns(3)
        with c1:
            org_group = sbox("Organisation Group", "ORG_GROUP", D["ORG_GROUP"])
        with c2:
            occupation = sbox("Occupation Type", "OCCUPATION_TYPE_GROUPED", D["OCCUPATION_TYPE_GROUPED"])
        with c3:
            n_docs = st.number_input("Documents Provided", min_value=0, max_value=20,
                                     value=int(D["N_DOCUMENTS_PROVIDED"]), step=1, key="w_ndocs")

    # ── Section 4: Credit & Region ───────────────────────────────────────────────
    st.markdown("### Credit Ratings & Region")
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            avg_ext_rating = st.slider("External Credit Rating (0=worst, 1=best)",
                                       0.0, 1.0, float(D["AVERAGE_EXTERNAL_RATING"]),
                                       step=0.001, format="%.3f", key="w_rating")
        with c2:
            region_rating = st.selectbox("Region Risk (1=best, 3=worst)", [1, 2, 3],
                                         index=[1,2,3].index(int(D["REGION_RATING_CLIENT"])),
                                         key="w_region")
        with c3:
            st.markdown("")  # spacer

    # ── Section 5: Previous Applications ─────────────────────────────────────────
    st.markdown("### Previous Application History")
    with st.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            tot_prev_app = st.number_input("Total Prev. Applications", min_value=0, max_value=50,
                                           value=int(D["TOT_PREV_APP"]), step=1, key="w_prevapp")
        with c2:
            approved_ratio = st.slider("Approved Ratio", 0.0, 1.0, float(D["APPROVED_RATIO"]), 0.01, key="w_apr")
        with c3:
            refused_ratio = st.slider("Refused Ratio", 0.0, 1.0, float(D["REFUSED_RATIO"]), 0.01, key="w_ref")
        with c4:
            cancelled_ratio = st.slider("Cancelled Ratio", 0.0, 1.0, float(D["CANCELLED_RATIO"]), 0.01, key="w_canc")
        with c5:
            unused_ratio = st.slider("Unused Ratio", 0.0, 1.0, float(D["UNUSED_RATIO"]), 0.01, key="w_unu")

    # Ratio warning
    ratio_sum = approved_ratio + refused_ratio + cancelled_ratio + unused_ratio
    if tot_prev_app > 0 and abs(ratio_sum - 1.0) > 0.06:
        st.warning(f"Ratios sum to {ratio_sum:.2f} — they should ideally sum to 1.0")

    # ══════════════════════════════════════════════════════════════════════════════
    # PREDICT BUTTON
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 3, 2])
    with btn_col:
        run = st.button("Run Credit Assessment")

    # ══════════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════════════════
    if run:
        input_data = {
            "CODE_GENDER":             gender,
            "FLAG_OWN_CAR":            own_car,
            "FLAG_OWN_REALTY":         own_realty,
            "CNT_CHILDREN":            cnt_children,
            "DEBT_TO_INCOME":          debt_to_income,
            "AVERAGE_EXTERNAL_RATING": avg_ext_rating,
            "N_DOCUMENTS_PROVIDED":    n_docs,
            "ADDITIONAL_DOC_PROVIDED": add_doc_val,
            "TOT_PREV_APP":            tot_prev_app,
            "APPROVED_RATIO":          approved_ratio,
            "REFUSED_RATIO":           refused_ratio,
            "CANCELLED_RATIO":         cancelled_ratio,
            "UNUSED_RATIO":            unused_ratio,
            "NAME_TYPE_SUITE":         type_suite,
            "NAME_INCOME_TYPE":        income_type,
            "NAME_FAMILY_STATUS":      family_status,
            "NAME_HOUSING_TYPE":       housing,
            "REGION_RATING_CLIENT":    region_rating,
            "YEARS_BIRTH":             years_birth,
            "YEARS_EMPLOYED":          years_employed,
            "YEARS_REGISTRATION":      years_registration,
            "YEARS_ID_PUBLISH":        years_id_publish,
            "YEARS_LAST_PHONE_CHANGE": years_phone,
            "ORG_GROUP":               org_group,
            "OCCUPATION_TYPE_GROUPED": occupation,
            "EDUCATION_LEVEL":         education,
        }
        df_input = pd.DataFrame([input_data])

        with st.spinner("Running model …"):
            try:
                model = load_model()
                prob  = float(model.predict_proba(df_input)[0, 1])
                pred  = int(model.predict(df_input)[0])
            except Exception as e:
                st.error(f"Model error: {e}")
                st.stop()

        # ── Verdict card ───────────────────────────────────────────────────────────
        st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)

        if prob >= 0.5:
            card_cls, verd_cls, verdict = "result-decline", "verdict-decline", "DECLINED"
        elif prob >= 0.25:
            card_cls, verd_cls, verdict = "result-review", "verdict-review", "REQUIRES REVIEW"
        else:
            card_cls, verd_cls, verdict = "result-approve", "verdict-approve", "APPROVED"

        st.markdown(f"""
        <div class="result-card {card_cls}">
            <div>
                <div class="verdict-text {verd_cls}">{verdict}</div>
                <div style="color:#94a3b8;font-size:0.9rem;margin-top:0.2rem;">
                    Default Probability: <b style="color:#e2e8f0">{prob:.1%}</b>
                    &nbsp;|&nbsp; Model Decision: <b style="color:#e2e8f0">{'BAD' if pred==1 else 'GOOD'}</b>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability bar ────────────────────────────────────────────────────────
        fig_bar, ax_bar = plt.subplots(figsize=(8, 0.55))
        fig_bar.patch.set_facecolor("#111827")
        ax_bar.set_facecolor("#111827")
        bar_color = "#ef4444" if prob >= 0.5 else ("#f59e0b" if prob >= 0.25 else "#22c55e")
        ax_bar.barh(0, prob,       color=bar_color,  height=0.6)
        ax_bar.barh(0, 1 - prob,   left=prob, color="#1e293b", height=0.6)
        ax_bar.set_xlim(0, 1); ax_bar.set_yticks([])
        ax_bar.tick_params(colors="#64748b", labelsize=8)
        ax_bar.spines[:].set_color("#1e293b")
        ax_bar.axvline(0.5, color="#475569", linestyle="--", linewidth=0.8)
        ax_bar.text(prob / 2, 0, f"{prob:.1%}", ha="center", va="center",
                    color="white", fontsize=8.5, fontweight="bold")
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

        # ── Pre-calculate SHAP for LLM context ─────────────────────────────────────
        shap_summary = ""
        shap_df_global = None
        try:
            import shap
            from xgboost import XGBClassifier as _XGB
            base = unwrap_xgboost(model)
            preproc = get_preproc_pipeline(model)
            if isinstance(base, _XGB):
                X_t = preproc.transform(df_input) if preproc else df_input.values
                try:
                    feat_names = preproc.get_feature_names_out() if preproc else list(df_input.columns)
                except:
                    feat_names = [f"f{i}" for i in range(X_t.shape[1])]
                
                explainer = shap.TreeExplainer(base)
                shap_vals = explainer.shap_values(X_t)
                
                # Target the last class (Class 1 = Default) for explanation
                if isinstance(shap_vals, list):
                    v = shap_vals[-1]
                else:
                    v = shap_vals
                
                # Global DF for plotting later
                shap_df_global = pd.DataFrame({
                    "Feature": feat_names,
                    "SHAP": v[0],
                }).reindex(pd.Series(v[0]).abs().sort_values(ascending=False).index).head(12)
                
                # String summary for LLM
                summary_lines = []
                for _, r in shap_df_global.head(8).iterrows():
                    direction = "increases risk" if r['SHAP'] > 0 else "reduces risk"
                    summary_lines.append(f"- {r['Feature']}: {r['SHAP']:.4f} ({direction})")
                shap_summary = "\n".join(summary_lines)
        except Exception as e:
            st.sidebar.warning(f"SHAP pre-calc failed: {e}")

        # ── Two-column layout for Report + SHAP ───────────────────────────────────
        col_rep, col_shap = st.columns([3, 2])

        with col_rep:
            st.markdown("""
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <h4 style="margin: 0;">AI Credit Report</h4>
                    <span style="background: rgba(102, 126, 234, 0.1); color: #667eea; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; border: 1px solid rgba(102, 126, 234, 0.2);">
                        Powered by GPT-4o-mini
                    </span>
                </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Writing narrative report..."):
                report_html = generate_report(input_data, prob, pred, shap_summary)
                st.markdown(report_html, unsafe_allow_html=True)

        with col_shap:
            st.markdown("#### Model Explanation (SHAP)")
            if shap_df_global is not None:
                fig_s, ax_s = plt.subplots(figsize=(5, 4.5))
                fig_s.patch.set_facecolor("#111827")
                ax_s.set_facecolor("#111827")
                colors = ["#ef4444" if v > 0 else "#22c55e" for v in shap_df_global["SHAP"]]
                ax_s.barh(shap_df_global["Feature"][::-1], shap_df_global["SHAP"][::-1], color=colors[::-1], height=0.6)
                ax_s.axvline(0, color="#475569", linewidth=0.8)
                ax_s.tick_params(colors="#94a3b8", labelsize=7.5)
                ax_s.spines[:].set_color("#1e293b")
                ax_s.set_xlabel("SHAP Value", color="#64748b", fontsize=8)
                ax_s.set_title("Red = increases risk  |  Green = reduces risk",
                                color="#64748b", fontsize=7.5, pad=8)
                plt.tight_layout()
                st.pyplot(fig_s, use_container_width=True)
                plt.close(fig_s)
            else:
                st.info("SHAP data not available for this model type.")

# ══════════════════════════════════════════════════════════════════════════════
# MONITORING TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_monitor:
    st.markdown("### Model Training Framework")
    col_fw1, col_fw2 = st.columns([3, 2])
    
    with col_fw1:
        st.markdown("""
        <div class="report-box" style="margin-bottom:2rem;">
        <b>Unified ML Pipeline Architecture:</b><br><br>
        1. <b>Data Ingestion:</b> Raw features (26 total) from application & credit history.<br>
        2. <b>Preprocessing:</b> 
           - <i>Log-scaling</i> for skewed financials.
           - <i>Standardization</i> for linear models.
           - <i>One-Hot Encoding</i> for categorical features.<br>
        3. <b>Sampling (Class Imbalance):</b> 
           - Either <b>Weighted Class Keys</b> or <b>SMOTE / RandomUnderSampler</b>.
           - <i>Logic:</i> Only use weights OR samplers, never both to avoid double-weighting.<br>
        4. <b>Model Engine:</b> XGBoost / Logistic Regression / Random Forest.<br>
        5. <b>Calibration:</b> Isotonic calibration applied to UnderSampled models to fix probability distortion.<br>
        6. <b>Optimization:</b> RandomizedSearchCV on Average Precision (PR-AUC).
        </div>
        """, unsafe_allow_html=True)

    with col_fw2:
        st.markdown("""
        <div class="section-card" style="font-size:0.8rem;">
        <b>Calibration Strategy:</b><br>
        Undersampling throws off the "odds". If we remove 90% of good cases, the model thinks defaults are 10x more likely. 
        <br><br>
        <b>Solution:</b> <i>CalibratedClassifierCV</i> recalibrates these probabilities back to real-world frequencies.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Experiment Results — Train vs Test")
    
    col_m1, col_m2 = st.columns([7, 3])
    with col_m1:
        st.markdown(
            "<div style='color:#64748b;font-size:0.85rem;margin-bottom:1rem;'>"
            "Metrics are pulled live from your local MLflow tracking server. "
            "Higher Average Precision = better. Watch for large Train–Test gaps (overfitting)."
            "</div>", unsafe_allow_html=True)
    with col_m2:
        st.link_button("Open MLflow Dashboard", "http://127.0.0.1:5000/", use_container_width=True)
    try:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        client = MlflowClient()

        # Search across all experiments
        all_exps = client.search_experiments()
        exp_ids  = [e.experiment_id for e in all_exps if e.name == "Credit Risk Modelling Project v6"]

        if not exp_ids:
            st.info("No experiments found. Run your notebook to log results.")
        else:
            runs = client.search_runs(
                experiment_ids=exp_ids,
                order_by=["metrics.test_avg_precision DESC"],
                max_results=20,
            )

            rows = []
            for r in runs:
                m    = r.data.metrics
                name = r.data.tags.get("mlflow.runName", r.info.run_id[:8])
                train_ap  = m.get("train_avg_precision", m.get("average_precision_score", None))
                test_ap   = m.get("test_avg_precision",  None)
                test_prec = m.get("test_precision",      None)
                test_rec  = m.get("test_recall",         None)
                test_f1   = m.get("test_f1",             None)
                f1_thresh = m.get("f1_threshold",        None)
                train_f1  = m.get("train_f1",            None)
                train_rec = m.get("train_recall",        None)
                train_prec = m.get("train_precision",     None)
                
                if test_ap is None:
                    continue   # skip runs with no useful metrics
                gap = round(train_ap - test_ap, 3) if train_ap is not None else None
                # Deep link to MLflow run
                run_url = f"http://127.0.0.1:5000/#/experiments/{exp_ids[0]}/runs/{r.info.run_id}"
                
                rows.append({
                    "Model": name,
                    "Train AP": round(train_ap, 3) if train_ap is not None else None,
                    "Test AP":  round(test_ap,  3),
                    "AP Gap":   gap,
                    "Test Precision": round(test_prec, 3) if test_prec is not None else None,
                    "Test Recall":    round(test_rec,  3) if test_rec  is not None else None,
                    "Test F1":        round(test_f1,   3) if test_f1   is not None else None,
                    "F1 Threshold":   round(f1_thresh, 3) if f1_thresh is not None else None,
                    "Train F1":       round(train_f1,  3) if train_f1  is not None else None,
                    "Train Recall":   round(train_rec, 3) if train_rec is not None else None,
                    "Train Precision": round(train_prec, 3) if train_prec is not None else None,
                    "View": f"[Inspect]({run_url})",
                    "Run ID": r.info.run_id[:8],
                })

            if not rows:
                st.info("Runs found but no metrics logged yet. Check your notebook.")
            else:
                df_m = pd.DataFrame(rows)

                # ── Force all metric columns to float64 so Arrow never sees object dtype ──
                _metric_cols = [
                    "Train AP", "Test AP", "AP Gap",
                    "Test Precision", "Test Recall", "Test F1", "F1 Threshold",
                    "Train F1", "Train Recall", "Train Precision"
                ]
                for _c in _metric_cols:
                    if _c in df_m.columns:
                        df_m[_c] = pd.to_numeric(df_m[_c], errors="coerce")

                # Colour-code the AP Gap column
                def colour_gap(val):
                    if not isinstance(val, (int, float)) or pd.isna(val):
                        return ""
                    if val > 0.15:
                        return "color: #f87171"   # red  – big overfit
                    elif val > 0.05:
                        return "color: #fbbf24"   # amber
                    return "color: #4ade80"        # green – healthy

                st.dataframe(
                    df_m.style
                    .map(colour_gap, subset=["AP Gap"])
                    .format(precision=3, na_rep="—"),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "View": st.column_config.LinkColumn(
                            "View",
                            help="Open this specific run in MLflow UI",
                            validate=r"^http://",
                            display_text="Inspect ↗"
                        )
                    }
                )

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Grouped bar chart: Train AP vs Test AP ───────────────────
                st.markdown("#### Average Precision: Train vs Test (top models)")
                df_plot = df_m.dropna(subset=["Test AP"]).copy()

                x      = range(len(df_plot))
                labels = df_plot["Model"].tolist()
                width  = 0.35

                fig, ax = plt.subplots(figsize=(10, max(3.5, len(df_plot) * 0.55)))
                fig.patch.set_facecolor("#0a0e1a")
                ax.set_facecolor("#111827")

                bars_tr = ax.barh([i - width/2 for i in x], df_plot["Train AP"].fillna(0),
                                  width, label="Train AP", color="#667eea", alpha=0.85)
                bars_te = ax.barh([i + width/2 for i in x], df_plot["Test AP"],
                                  width, label="Test AP",  color="#f472b6", alpha=0.85)

                ax.set_yticks(list(x))
                ax.set_yticklabels(labels, fontsize=7.5, color="#94a3b8")
                ax.tick_params(colors="#64748b", labelsize=8)
                ax.spines[:].set_color("#2d3748")
                ax.set_xlabel("Average Precision (PR-AUC)", color="#64748b", fontsize=8)
                ax.legend(fontsize=8, facecolor="#1e293b", labelcolor="#e2e8f0",
                          edgecolor="#2d3748")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # ── F1 score comparison ──────────────────────────────────────
                st.markdown("#### F1 Score: Train vs Test")
                df_f1 = df_m.copy()
                df_f1["Test F1"]  = pd.to_numeric(df_f1["Test F1"], errors="coerce")
                df_f1["Train F1"] = pd.to_numeric(df_f1["Train F1"], errors="coerce")
                df_f1 = df_f1.dropna(subset=["Test F1"]).sort_values("Test F1").tail(8)

                if not df_f1.empty:
                    x2 = range(len(df_f1))
                    fig2, ax2 = plt.subplots(figsize=(10, max(3, len(df_f1) * 0.5)))
                    fig2.patch.set_facecolor("#0a0e1a")
                    ax2.set_facecolor("#111827")
                    ax2.barh([i - width/2 for i in x2], df_f1["Train F1"].fillna(0),
                             width, label="Train F1", color="#34d399", alpha=0.85)
                    ax2.barh([i + width/2 for i in x2], df_f1["Test F1"],
                             width, label="Test F1",  color="#fb923c", alpha=0.85)
                    ax2.set_yticks(list(x2))
                    ax2.set_yticklabels(df_f1["Model"].tolist(), fontsize=7.5, color="#94a3b8")
                    ax2.tick_params(colors="#64748b", labelsize=8)
                    ax2.spines[:].set_color("#2d3748")
                    ax2.set_xlabel("F1 Score", color="#64748b", fontsize=8)
                    ax2.legend(fontsize=8, facecolor="#1e293b", labelcolor="#e2e8f0",
                               edgecolor="#2d3748")
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)

                # ── Key insight callout ──────────────────────────────────────
                best = df_m.iloc[0] if not df_m.empty else None
                if best is not None:
                    st.markdown(f"""
                    <div class="section-card" style="margin-top:1.5rem;">
                    <b>Recommended Production Model:</b>&nbsp;
                    <span style="color:#667eea;font-weight:600;">{best['Model']}</span>
                    &nbsp;—&nbsp; Test AP: <b>{best['Test AP']}</b>,
                    Test F1: <b>{best['Test F1']}</b>,
                    Test Recall: <b>{best['Test Recall']}</b>
                    <br><span style="color:#64748b;font-size:0.8rem;">
                    The AP Gap (Train – Test) of <b>{best['AP Gap']}</b> indicates
                    {'good generalisation' if isinstance(best['AP Gap'], float) and best['AP Gap'] < 0.1 else 'some overfitting — monitor closely'}.
                    </span>
                    </div>
                    """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Could not load MLflow metrics: {e}")

