"""
Report Service - Handles AI-powered report generation with OpenAI and RAG
"""
import json
import os
import streamlit as st
from typing import Optional
from openai import OpenAI

from ..config.settings import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MAX_TOKENS, OPENAI_TEMPERATURE,
    RISK_THRESHOLD_DEFAULT, DOCUMENTS_DIR, NB_DIR
)


class ReportService:
    """Service for generating credit risk reports"""
    
    def __init__(self):
        self._client = None
        self._rag_context = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client if API key is available"""
        if OPENAI_API_KEY:
            self._client = OpenAI(api_key=OPENAI_API_KEY)
    
    @property
    def is_ai_available(self) -> bool:
        """Check if AI client is available"""
        return self._client is not None
    
    def generate_report(self, data: dict, prob: float, pred: int, 
                       shap_summary: str = "") -> str:
        """
        Generate credit risk report
        
        Args:
            data: Applicant data dictionary
            prob: Probability of default
            pred: Model prediction (0 or 1)
            shap_summary: SHAP feature importance summary
            use_rag: Whether to use RAG for enhanced context
            
        Returns:
            HTML formatted report string
        """
        if self.is_ai_available:
            return self._generate_llm_report(data, prob, pred, shap_summary)
        return self._generate_legacy_report(data, prob, pred)
    
    def _get_rag_context(self) -> str:
        """Load RAG context from project documentation for internal reports"""
        if self._rag_context is not None:
            return self._rag_context
        
        context_parts = []
        
        # Load model training info
        try:
            model_info = """
MODEL TRAINING CONTEXT:
- Dataset: Home Credit Default Risk dataset with 307,511 applications
- Features: 26 engineered features from application and previous loan history
- Target: Binary classification (1 = Default, 0 = No Default)
- Class Imbalance: ~8% default rate, handled via SMOTE/Undersampling + Calibration
- Model: XGBoost with isotonic calibration for probability correction
- Optimization: RandomizedSearchCV on Average Precision (PR-AUC)
- Validation: Stratified K-Fold cross-validation

FEATURE ENGINEERING:
- DEBT_TO_INCOME: Credit amount / Annual income
- AVERAGE_EXTERNAL_RATING: Mean of external credit bureau scores (normalized 0-1)
- YEARS_EMPLOYED: Employment duration (negative = unemployed/pensioner)
- Previous application ratios: APPROVED_RATIO, REFUSED_RATIO, CANCELLED_RATIO
- Document flags aggregated into N_DOCUMENTS_PROVIDED

BUSINESS RULES:
- PD >= 50%: DECLINE (High Risk)
- PD 25-50%: REVIEW (Medium Risk - requires manual assessment)
- PD < 25%: APPROVE (Low Risk)

RISK FACTORS (from SHAP analysis):
- High DEBT_TO_INCOME strongly increases default probability
- Low AVERAGE_EXTERNAL_RATING is a major risk indicator
- Short employment history (YEARS_EMPLOYED < 1) increases risk
- High REFUSED_RATIO from previous applications is a red flag
- Region rating 3 (highest risk region) adds to default probability
"""
            context_parts.append(model_info)
        except Exception:
            pass
        
        self._rag_context = "\n".join(context_parts)
        return self._rag_context
    
    def _generate_llm_report(self, data: dict, prob: float, pred: int, 
                            shap_summary: str) -> str:
        """Generate report using OpenAI GPT"""
        try:
            model_decision = self._get_decision(prob)
            
            prompt = f"""
You are a senior credit risk analyst generating a professional credit assessment report.

You are a senior credit risk analyst generating a professional credit assessment report.

MODEL PREDICTION:
- Predicted Probability of Default (PD): {prob:.1%}
- Verdict: {model_decision}

APPLICANT DATA:
{json.dumps(data, indent=2)}

SHAP EXPLANATION (Feature Contributions):
{shap_summary if shap_summary else "Not available"}

Generate a structured credit risk report with these sections:

1. Executive Summary - Brief overview of the assessment outcome
2. Risk Analysis - Key risk factors identified from the data and SHAP values
   - POSITIVE SHAP = increases default risk
   - NEGATIVE SHAP = reduces default risk
3. Applicant Profile Strengths - Positive aspects of the application
4. Areas of Concern - Specific red flags requiring attention
5. Recommendations
   - For the Customer: Steps to improve creditworthiness
   - For the Bank: Due diligence suggestions and conditions
6. Decision Rationale - Why this verdict was reached

IMPORTANT: Output ONLY raw HTML. Use <b> for bold, <br> for line breaks, <ul><li> for lists.
Do NOT escape HTML entities. Do NOT use markdown. Output the HTML directly as if writing an HTML file.
Start directly with the content - no preamble or code blocks.
"""
            response = self._client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst. Output raw HTML only - no markdown, no escaped entities, no code blocks."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=OPENAI_MAX_TOKENS,
                temperature=OPENAI_TEMPERATURE
            )
            content = response.choices[0].message.content.strip()
            
            # Remove code block markers if AI added them
            if content.startswith("```html"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            return f'<div class="report-box">{content}</div>'
            
        except Exception as e:
            return self._generate_legacy_report(data, prob, pred, error=str(e))
    
    def _generate_legacy_report(self, data: dict, prob: float, pred: int, 
                               error: Optional[str] = None) -> str:
        """Generate rule-based fallback report"""
        thresh = st.session_state.get("current_threshold", RISK_THRESHOLD_DEFAULT)
        review_thresh = thresh * 0.8
        
        verdict = "HIGH RISK" if prob >= thresh else (
            "MEDIUM RISK" if prob >= review_thresh else "LOW RISK"
        )
        rec = "decline" if prob >= thresh else (
            "review" if prob >= review_thresh else "approve"
        )
        
        flags = []
        if data.get("DEBT_TO_INCOME", 0) > 5:
            flags.append(f"High debt-to-income ratio: {data['DEBT_TO_INCOME']:.1f}")
        if data.get("AVERAGE_EXTERNAL_RATING", 1) < 0.3:
            flags.append(f"Low external credit rating: {data['AVERAGE_EXTERNAL_RATING']:.2f}")
        if data.get("REFUSED_RATIO", 0) > 0.4:
            flags.append(f"High previous refusal rate: {data['REFUSED_RATIO']:.0%}")
        if data.get("YEARS_EMPLOYED", 10) < 1:
            flags.append("Very short employment history (under 1 year)")
        if data.get("REGION_RATING_CLIENT") == 3:
            flags.append("High-risk region rating (3/3)")
        
        strengths = []
        if data.get("FLAG_OWN_REALTY") == 1:
            strengths.append("Owns real estate")
        if data.get("AVERAGE_EXTERNAL_RATING", 0) >= 0.7:
            strengths.append(f"Strong external rating: {data['AVERAGE_EXTERNAL_RATING']:.2f}")
        if data.get("YEARS_EMPLOYED", 0) >= 5:
            strengths.append(f"Stable employment: {data['YEARS_EMPLOYED']:.1f} years")
        if data.get("APPROVED_RATIO", 0) >= 0.7:
            strengths.append(f"Good approval history: {data['APPROVED_RATIO']:.0%}")
        
        error_note = f"<br><small style='color:#f59e0b;'>⚠️ AI report unavailable: {error}</small>" if error else ""
        
        report = f"""
<div class="report-box">
<b>Executive Summary</b><br>
This application has been assessed as <b>{verdict}</b> with a probability of default of <b>{prob:.1%}</b>. 
The recommendation is to <b>{rec}</b> this loan request.
{error_note}
<br><br>
<b>Risk Factors Identified:</b><br>
{"<ul>" + "".join(f"<li>{f}</li>" for f in flags) + "</ul>" if flags else "No major red flags identified."}
<br>
<b>Applicant Strengths:</b><br>
{"<ul>" + "".join(f"<li>{s}</li>" for s in strengths) + "</ul>" if strengths else "Limited positive indicators."}
<br>
<b>Recommendation:</b><br>
{"Proceed with standard approval process." if rec == "approve" else 
 "Requires manual review by credit committee." if rec == "review" else
 "Application does not meet risk criteria. Consider declining or requesting additional collateral."}
</div>
"""
        return report
    
    def _get_decision(self, prob: float) -> str:
        """Get decision string based on probability"""
        thresh = st.session_state.get("current_threshold", RISK_THRESHOLD_DEFAULT)
        review_thresh = thresh * 0.8
        if prob >= thresh:
            return "DECLINE"
        elif prob >= review_thresh:
            return "REVIEW"
        return "APPROVE"
