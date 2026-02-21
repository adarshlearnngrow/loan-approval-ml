"""
Credit Risk Predictor — Premium Streamlit Application (Modular Version)
========================================================================
Run from the project root:
    streamlit run app.py
"""

import sys
import pathlib

# Add src to path
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config.settings import APP_TITLE, APP_SUBTITLE, APP_VERSION  # APP_VERSION used in header badge
from src.components.styles import inject_custom_css
from src.components.input_form import render_input_form
from src.components.results import render_results
from src.components.monitoring import render_monitoring_tab
from src.services.model_service import ModelService
from src.services.report_service import ReportService

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="",
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
    <h1>{APP_TITLE}</h1>
    <div class="subtitle">{APP_SUBTITLE}</div>
  </div>
  <div class="version-badge">v{APP_VERSION}</div>
</div>
""", unsafe_allow_html=True)

# ── Main Tabs ──────────────────────────────────────────────────────────────────
tab_assess, tab_monitor = st.tabs([
    "Risk Assessment",
    "Model Monitoring",
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

    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        run = st.button("Run Credit Assessment", use_container_width=True)

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
                input_data, prob, pred, shap_summary
            )

        st.markdown(
            f"<div style='color:#64748b;font-size:0.78rem;margin-bottom:0.5rem;'>"
            f"Decision threshold: <b style='color:#94a3b8;'>{model_service.threshold:.3f}</b>"
            f" &nbsp;(sourced from MLflow <code>f1_threshold</code>)"
            f"</div>",
            unsafe_allow_html=True,
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

