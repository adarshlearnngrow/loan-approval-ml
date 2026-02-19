"""
Credit Risk Predictor — Premium Streamlit Application (Modular Version)
========================================================================
Run from the project root:
    streamlit run app_modular.py
"""

import sys
import pathlib

# Add src to path
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config.settings import APP_TITLE, APP_SUBTITLE, APP_VERSION
from src.components.styles import inject_custom_css
from src.components.input_form import render_input_form
from src.components.results import render_results
from src.components.monitoring import render_monitoring_tab
from src.services.model_service import ModelService
from src.services.report_service import ReportService

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inject custom CSS ──────────────────────────────────────────────────────────
inject_custom_css()

# ── Initialize services ────────────────────────────────────────────────────────
model_service = ModelService()
report_service = ReportService()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="header-bar">
  <div>
    <h1>🏦 {APP_TITLE}</h1>
    <div class="subtitle">{APP_SUBTITLE}</div>
  </div>
  <div class="version-badge">v{APP_VERSION}</div>
</div>
""", unsafe_allow_html=True)

# ── Main Tabs ──────────────────────────────────────────────────────────────────
tab_assess, tab_monitor, tab_about = st.tabs([
    "🎯 Risk Assessment", 
    "📊 Model Monitoring",
    "ℹ️ About"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: RISK ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
with tab_assess:
    # Get categories for form
    categories = model_service.get_categories()
    
    # Render input form
    input_data = render_input_form(categories)
    
    # Predict button
    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        # RAG toggle for internal team
        use_rag = st.checkbox(
            "🔒 Use RAG Context (Internal Team)", 
            value=False,
            help="Enable RAG-based report generation with internal model documentation"
        )
        run = st.button("🚀 Run Credit Assessment", use_container_width=True)
        # Show the threshold that was loaded from MLflow
        loaded_threshold = model_service.threshold
        st.markdown(
            f"<div style='text-align:center;color:#64748b;font-size:0.78rem;margin-top:0.4rem;'>"
            f"Decision threshold: <b style='color:#94a3b8;'>{loaded_threshold:.3f}</b>"
            f" &nbsp;(sourced from MLflow <code>f1_threshold</code>)"
            f"</div>",
            unsafe_allow_html=True,
        )
    
    # ── Results ────────────────────────────────────────────────────────────────
    if run:
        with st.spinner("Running model prediction..."):
            try:
                prob, pred, shap_summary, shap_df = model_service.predict(input_data)
            except Exception as e:
                st.error(f"Model error: {e}")
                st.stop()
        
        with st.spinner("Generating credit report..."):
            report_html = report_service.generate_report(
                input_data, prob, pred, shap_summary, use_rag=use_rag
            )
        
        render_results(
            prob=prob,
            pred=pred,
            report_html=report_html,
            shap_df=shap_df,
            is_ai_report=report_service.is_ai_available
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: MODEL MONITORING
# ══════════════════════════════════════════════════════════════════════════════
with tab_monitor:
    render_monitoring_tab()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### About This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="section-card">
        <b>🎯 Purpose</b><br><br>
        This Credit Risk Predictor is an AI-powered decision support system designed to:
        <ul>
            <li>Assess loan application risk in real-time</li>
            <li>Provide explainable AI insights via SHAP</li>
            <li>Generate professional credit reports</li>
            <li>Monitor model performance over time</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="section-card">
        <b>🔧 Technology Stack</b><br><br>
        <ul>
            <li><b>ML Framework:</b> XGBoost with Isotonic Calibration</li>
            <li><b>Experiment Tracking:</b> MLflow</li>
            <li><b>Explainability:</b> SHAP (TreeExplainer)</li>
            <li><b>Report Generation:</b> OpenAI GPT-4o</li>
            <li><b>Frontend:</b> Streamlit</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="section-card">
        <b>⚠️ Important Disclaimer</b><br><br>
        This tool is designed as a <b>decision support system</b>, not a replacement 
        for human judgment. All credit decisions should be reviewed by qualified 
        personnel before final approval.
        <br><br>
        The model predictions are based on historical data patterns and may not 
        capture all relevant factors for individual cases.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="section-card">
        <b>📋 Version Information</b><br><br>
        <ul>
            <li><b>Application Version:</b> {APP_VERSION}</li>
            <li><b>Model:</b> Credit Risk Model Final</li>
            <li><b>AI Reports:</b> {'✅ Enabled' if report_service.is_ai_available else '❌ Disabled (No API Key)'}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
