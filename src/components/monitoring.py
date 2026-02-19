"""
Monitoring Component - Renders model monitoring dashboard
"""
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..services.mlflow_service import MLflowService
from ..config.settings import MLFLOW_TRACKING_URI


def render_monitoring_tab():
    """Render the model monitoring tab content"""
    
    # Framework overview
    _render_framework_overview()
    
    st.markdown("### Experiment Results — Train vs Test")
    
    col_m1, col_m2 = st.columns([7, 3])
    with col_m1:
        st.markdown(
            "<div style='color:#64748b;font-size:0.85rem;margin-bottom:1rem;'>"
            "Metrics are pulled live from your local MLflow tracking server. "
            "Higher Average Precision = better. Watch for large Train–Test gaps (overfitting)."
            "</div>", unsafe_allow_html=True
        )
    with col_m2:
        st.link_button("Open MLflow Dashboard", MLFLOW_TRACKING_URI, use_container_width=True)
    
    try:
        mlflow_service = MLflowService()
        df_metrics = mlflow_service.get_experiment_runs()
        
        if df_metrics.empty:
            st.info("No experiments found. Run your notebook to log results.")
            return
        
        # Display metrics table
        _render_metrics_table(df_metrics)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts
        _render_ap_comparison_chart(df_metrics)
        _render_f1_comparison_chart(df_metrics)
        
        # Best model callout
        best_info = mlflow_service.get_best_model_info(df_metrics)
        if best_info:
            _render_best_model_callout(best_info)
            
    except Exception as e:
        st.error(f"Could not load MLflow metrics: {e}")


def _render_framework_overview():
    """Render the ML pipeline framework overview"""
    st.markdown("### Model Training Framework")
    
    col_fw1, col_fw2 = st.columns([3, 2])
    
    with col_fw1:
        st.markdown("""
        <div class="report-box" style="margin-bottom:2rem;">
        <b>Unified ML Pipeline Architecture:</b><br><br>
        <b>1. Data Ingestion:</b> Raw features (26 total) from application & credit history.<br><br>
        <b>2. Preprocessing:</b><br>
           • <i>Log-scaling</i> for skewed financials<br>
           • <i>Standardization</i> for linear models<br>
           • <i>One-Hot Encoding</i> for categorical features<br><br>
        <b>3. Sampling (Class Imbalance):</b><br>
           • <b>Weighted Class Keys</b> or <b>SMOTE / RandomUnderSampler</b><br>
           • Only use weights OR samplers, never both<br><br>
        <b>4. Model Engine:</b> XGBoost / Logistic Regression / Random Forest<br><br>
        <b>5. Calibration:</b> Isotonic calibration for UnderSampled models<br><br>
        <b>6. Optimization:</b> RandomizedSearchCV on Average Precision (PR-AUC)
        </div>
        """, unsafe_allow_html=True)
    
    with col_fw2:
        st.markdown("""
        <div class="section-card" style="font-size:0.85rem;">
        <b>Calibration Strategy:</b><br><br>
        Undersampling distorts class probabilities. If we remove 90% of good cases, 
        the model thinks defaults are 10x more likely.
        <br><br>
        <b>Solution:</b> <i>CalibratedClassifierCV</i> with isotonic regression 
        recalibrates probabilities back to real-world frequencies.
        <br><br>
        <b>Why it matters:</b> Accurate probabilities enable proper risk-based pricing 
        and regulatory compliance (Basel III/IV).
        </div>
        """, unsafe_allow_html=True)


def _render_metrics_table(df: pd.DataFrame):
    """Render the metrics comparison table"""
    
    def colour_gap(val):
        if not isinstance(val, (int, float)) or pd.isna(val):
            return ""
        if val > 0.15:
            return "color: #f87171"  # red - big overfit
        elif val > 0.05:
            return "color: #fbbf24"  # amber
        return "color: #4ade80"      # green - healthy
    
    st.dataframe(
        df.style
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


def _render_ap_comparison_chart(df: pd.DataFrame):
    """Render Average Precision comparison chart"""
    st.markdown("#### Average Precision: Train vs Test")
    
    df_plot = df.dropna(subset=["Test AP"]).copy()
    if df_plot.empty:
        return
    
    x = range(len(df_plot))
    labels = df_plot["Model"].tolist()
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, max(3.5, len(df_plot) * 0.55)))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#111827")
    
    ax.barh([i - width/2 for i in x], df_plot["Train AP"].fillna(0),
            width, label="Train AP", color="#667eea", alpha=0.85)
    ax.barh([i + width/2 for i in x], df_plot["Test AP"],
            width, label="Test AP", color="#f472b6", alpha=0.85)
    
    ax.set_yticks(list(x))
    ax.set_yticklabels(labels, fontsize=7.5, color="#94a3b8")
    ax.tick_params(colors="#64748b", labelsize=8)
    ax.spines[:].set_color("#2d3748")
    ax.set_xlabel("Average Precision (PR-AUC)", color="#64748b", fontsize=8)
    ax.legend(fontsize=8, facecolor="#1e293b", labelcolor="#e2e8f0", edgecolor="#2d3748")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_f1_comparison_chart(df: pd.DataFrame):
    """Render F1 Score comparison chart"""
    st.markdown("#### F1 Score: Train vs Test")
    
    df_f1 = df.copy()
    df_f1["Test F1"] = pd.to_numeric(df_f1["Test F1"], errors="coerce")
    df_f1["Train F1"] = pd.to_numeric(df_f1["Train F1"], errors="coerce")
    df_f1 = df_f1.dropna(subset=["Test F1"]).sort_values("Test F1").tail(8)
    
    if df_f1.empty:
        return
    
    x = range(len(df_f1))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, max(3, len(df_f1) * 0.5)))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#111827")
    
    ax.barh([i - width/2 for i in x], df_f1["Train F1"].fillna(0),
            width, label="Train F1", color="#34d399", alpha=0.85)
    ax.barh([i + width/2 for i in x], df_f1["Test F1"],
            width, label="Test F1", color="#fb923c", alpha=0.85)
    
    ax.set_yticks(list(x))
    ax.set_yticklabels(df_f1["Model"].tolist(), fontsize=7.5, color="#94a3b8")
    ax.tick_params(colors="#64748b", labelsize=8)
    ax.spines[:].set_color("#2d3748")
    ax.set_xlabel("F1 Score", color="#64748b", fontsize=8)
    ax.legend(fontsize=8, facecolor="#1e293b", labelcolor="#e2e8f0", edgecolor="#2d3748")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_best_model_callout(info: dict):
    """Render the best model recommendation callout"""
    gen_status = "good generalisation" if info["generalization"] == "good" else "some overfitting — monitor closely"
    
    st.markdown(f"""
    <div class="section-card" style="margin-top:1.5rem;">
    <b>Recommended Production Model:</b>&nbsp;
    <span style="color:#667eea;font-weight:600;">{info['name']}</span>
    &nbsp;—&nbsp; Test AP: <b>{info['test_ap']}</b>,
    Test F1: <b>{info.get('test_f1', 'N/A')}</b>,
    Test Recall: <b>{info.get('test_recall', 'N/A')}</b>
    <br><span style="color:#64748b;font-size:0.8rem;">
    The AP Gap (Train – Test) of <b>{info['ap_gap']}</b> indicates {gen_status}.
    </span>
    </div>
    """, unsafe_allow_html=True)
