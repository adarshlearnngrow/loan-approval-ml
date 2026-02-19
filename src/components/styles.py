"""
Premium Dark Theme CSS Styles
"""
import streamlit as st


def inject_custom_css():
    """Inject premium dark theme CSS into Streamlit app"""
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #0a0e1a 100%);
    color: #e2e8f0;
}

/* Hide sidebar toggle */
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"] { display: none; }

/* Main container */
.block-container { 
    max-width: 1200px; 
    padding: 2rem 2rem 4rem 2rem; 
}

/* ── Typography ─────────────────────────────────────────────────────────── */
h1 { 
    font-size: 2.2rem !important; 
    font-weight: 700 !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f472b6 100%);
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent;
    margin-bottom: 0 !important;
    text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
}

h3 { 
    color: #94a3b8 !important; 
    font-weight: 600 !important;
    font-size: 0.9rem !important; 
    text-transform: uppercase;
    letter-spacing: 1.5px; 
    margin-top: 2rem !important;
    margin-bottom: 0.8rem !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

h3::before {
    content: '';
    width: 4px;
    height: 16px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 2px;
}

h4 {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    margin-bottom: 0.8rem !important;
}

/* ── Input widgets ──────────────────────────────────────────────────────── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
div[data-baseweb="select"] > div {
    background: rgba(19, 25, 41, 0.8) !important;
    border: 1px solid rgba(45, 55, 72, 0.6) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

div[data-baseweb="select"] > div:hover,
[data-testid="stNumberInput"] input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 20px rgba(102, 126, 234, 0.2) !important;
}

[data-testid="stSlider"] { padding: 0; }
.stSlider [data-baseweb="slider"] { 
    background: rgba(19, 25, 41, 0.8); 
}

/* ── Section card ───────────────────────────────────────────────────────── */
.section-card {
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid rgba(30, 41, 59, 0.5);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.4rem;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.section-card:hover {
    border-color: rgba(102, 126, 234, 0.3);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important; 
    border: none !important;
    border-radius: 12px !important; 
    font-weight: 600 !important;
    font-size: 1rem !important; 
    padding: 0.8rem 2.5rem !important;
    letter-spacing: 0.5px;
    transition: all 0.3s ease !important;
    width: 100%;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.stButton > button:hover { 
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result cards ───────────────────────────────────────────────────────── */
.result-card {
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
    border: 1px solid;
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}

.result-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
}

.result-approve {
    background: linear-gradient(135deg, rgba(15, 36, 23, 0.9), rgba(13, 31, 18, 0.9));
    border-color: rgba(34, 197, 94, 0.3);
}
.result-approve::before {
    background: linear-gradient(90deg, #22c55e, #4ade80);
}

.result-review {
    background: linear-gradient(135deg, rgba(28, 26, 15, 0.9), rgba(26, 24, 0, 0.9));
    border-color: rgba(245, 158, 11, 0.3);
}
.result-review::before {
    background: linear-gradient(90deg, #f59e0b, #fbbf24);
}

.result-decline {
    background: linear-gradient(135deg, rgba(31, 15, 15, 0.9), rgba(26, 10, 10, 0.9));
    border-color: rgba(239, 68, 68, 0.3);
}
.result-decline::before {
    background: linear-gradient(90deg, #ef4444, #f87171);
}

.verdict-text {
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}

.verdict-approve { 
    color: #4ade80; 
    text-shadow: 0 0 20px rgba(74, 222, 128, 0.3);
}
.verdict-review { 
    color: #fbbf24; 
    text-shadow: 0 0 20px rgba(251, 191, 36, 0.3);
}
.verdict-decline { 
    color: #f87171; 
    text-shadow: 0 0 20px rgba(248, 113, 113, 0.3);
}

/* ── Report box ─────────────────────────────────────────────────────────── */
.report-box {
    background: rgba(19, 25, 41, 0.8);
    border: 1px solid rgba(45, 55, 72, 0.5);
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    font-size: 0.95rem;
    line-height: 1.9;
    color: #cbd5e1;
    backdrop-filter: blur(10px);
}

.report-box b { 
    color: #e2e8f0; 
    font-weight: 600;
}

.report-box ul {
    margin: 0.5rem 0;
    padding-left: 1.2rem;
}

.report-box li {
    margin: 0.3rem 0;
}

/* ── Divider ────────────────────────────────────────────────────────────── */
.thin-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
    margin: 2.5rem 0;
}

/* ── Header bar ─────────────────────────────────────────────────────────── */
.header-bar {
    display: flex; 
    align-items: center; 
    justify-content: space-between;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(45, 55, 72, 0.3);
}

.subtitle { 
    color: #64748b; 
    font-size: 0.95rem; 
    margin-top: 0.5rem;
    font-weight: 400;
}

/* ── Version badge ──────────────────────────────────────────────────────── */
.version-badge {
    background: rgba(102, 126, 234, 0.1);
    color: #667eea;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid rgba(102, 126, 234, 0.2);
}

/* ── AI badge ───────────────────────────────────────────────────────────── */
.ai-badge {
    background: rgba(102, 126, 234, 0.1);
    color: #667eea;
    padding: 4px 12px;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid rgba(102, 126, 234, 0.2);
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.ai-badge::before {
    content: '';
}

/* ── Metric cards ───────────────────────────────────────────────────────── */
.metric-card {
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid rgba(30, 41, 59, 0.5);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    color: #64748b;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

/* ── Tabs styling ───────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(17, 24, 39, 0.5);
    padding: 8px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
}

/* ── Quick fill button ──────────────────────────────────────────────────── */
.quick-fill-btn {
    background: rgba(102, 126, 234, 0.1) !important;
    border: 1px solid rgba(102, 126, 234, 0.3) !important;
    color: #667eea !important;
}

/* ── Animations ─────────────────────────────────────────────────────────── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.result-card {
    animation: fadeIn 0.5s ease-out;
}

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #0a0e1a;
}

::-webkit-scrollbar-thumb {
    background: #2d3748;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #4a5568;
}
</style>
""", unsafe_allow_html=True)
