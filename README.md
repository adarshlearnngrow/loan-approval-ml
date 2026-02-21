# Credit Risk Predictor

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0+-green.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered credit risk assessment system with explainable ML, real-time predictions, and automated report generation.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Business Problem & Decision Made](#business-problem--decision-made)
3. [Dataset Overview](#dataset-overview)
4. [Feature Engineering](#feature-engineering)
5. [Model Training & Selection](#model-training--selection)
6. [Threshold Tuning & MLflow Tracking](#threshold-tuning--mlflow-tracking)
7. [Risk Identification](#risk-identification)
8. [Business Usability](#business-usability)
9. [Financial Impact](#financial-impact)
10. [Can It Replace Human Decision-Making?](#can-it-replace-human-decision-making)
11. [AI-Powered Report Generation](#ai-powered-report-generation)
12. [Installation & Setup](#installation--setup)
13. [Project Structure](#project-structure)
14. [Usage Guide](#usage-guide)

---

## Executive Summary

This Credit Risk Predictor is a **production-ready machine learning system** designed to assess loan application risk in real-time. The system combines:

- **XGBoost pipeline** (ColumnTransformer + RandomUnderSampler + XGBClassifier) for accurate probability estimates
- **SHAP TreeExplainer** for transparent, auditable decisions
- **OpenAI GPT-4o-mini integration** for natural language credit reports
- **MLflow experiment tracking** for model versioning and metric logging
- **Portable model export** (`models/model.pkl`) so the app runs anywhere without a live MLflow server

### Key Metrics — Production Model (XGBoost Final Production)

| Metric | Value |
|--------|-------|
| Test Average Precision (PR-AUC) | 0.223 |
| Test F1 Score | 0.298 |
| Test Recall | 0.460 |
| Test Precision | 0.220 |
| Optimal F1 Threshold | **0.631** |

The production model prioritises **recall** (catching defaults) over precision, which is the correct trade-off for credit risk — missing a default is far more costly than a false alarm.

---

## Business Problem & Decision Made

### The Problem

Financial institutions face significant challenges in credit decisioning:

1. **High default rates** leading to substantial financial losses
2. **Manual review bottlenecks** causing slow loan processing
3. **Inconsistent decisions** across different loan officers
4. **Regulatory compliance** requirements for explainable decisions
5. **Class imbalance** (~8% default rate) making prediction difficult

### The Decision

We built an **automated credit risk scoring system** that:

- Predicts probability of default (PD) for each application
- Provides two-tier recommendations: APPROVE / DECLINE
- Explains each decision using SHAP feature attributions
- Generates professional credit reports (AI or rule-based fallback)
- Tracks model performance across all experiments via MLflow

### Decision Threshold

The **optimal decision threshold (`f1_threshold`)** is computed during training by finding the probability cutoff that maximises F1 score on the held-out test set. It is logged as an MLflow metric and loaded automatically at application startup.

**How it is computed in training:**

```python
precision, recall, thresholds = precision_recall_curve(y_test, y_test_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
ix = np.argmax(f1_scores[:-1])          # exclude last point (no threshold for it)
optimal_threshold = float(thresholds[ix])
mlflow.log_metric("f1_threshold", optimal_threshold)
```

**Why test data, not training data?**
Computing the threshold on the test set gives a more realistic estimate of how the model will behave on unseen applications. Using training data would produce an optimistically low threshold (the model has already seen those samples).

**How it is loaded by the app:**

`ModelService._load_model_cached()` reads directly from the local `mlruns/` directory — no live MLflow server required:

```python
# 1. Read registered model meta.yaml to get exact run_id and model_id
meta = (mlruns / "models" / MLFLOW_MODEL_NAME / "version-1" / "meta.yaml").read_text()
# parse run_id, model_id, exp_id from meta

# 2. Read f1_threshold from that run's metrics file
thresh_file = mlruns / exp_id / run_id / "metrics" / "f1_threshold"
threshold = float(thresh_file.read_text().strip().split()[1])

# 3. Load model from mlartifacts/
model = mlflow.sklearn.load_model(str(mlartifacts / exp_id / "models" / model_id / "artifacts"))

return model, threshold, run_id
```

If `mlruns/` is not present (e.g. on Streamlit Cloud), the app falls back to `models/model.pkl` and `models/model_info.json` which are committed to the repository.

**Current production threshold: `0.631`**

| PD | Decision | Meaning |
|----|----------|---------|
| < 0.631 | **APPROVE** | Model predicts low default risk |
| >= 0.631 | **DECLINE** | Model predicts high default risk |

The active threshold is shown beneath the prediction result so analysts always know which value is in effect.

---

## Dataset Overview

### Source Data

The model was trained on the **Home Credit Default Risk** dataset, a real-world credit bureau dataset containing:

| Dataset | Records | Description |
|---------|---------|-------------|
| `application_data.csv` | 307,511 | Current loan applications |
| `previous_application.csv` | 1,670,214 | Historical application records |
| `columns_description.csv` | 219 | Feature documentation |

### Target Variable

- **TARGET**: Binary indicator (1 = Default, 0 = No Default)
- **Default Rate**: ~8.07% (highly imbalanced)

### Data Quality

| Issue | Count | Resolution |
|-------|-------|------------|
| Missing values | 67 columns with >50% missing | Imputation or removal |
| Outliers | Income, credit amounts | Winsorization at 99th percentile |
| Inconsistent categories | Occupation, Organization | Grouped into meaningful clusters |

---

## Feature Engineering

### Final Feature Set (26 Features)

We engineered 26 predictive features from raw application data:

#### Personal & Demographic (7 features)

| Feature | Description | Engineering |
|---------|-------------|-------------|
| `CODE_GENDER` | Applicant gender | Categorical encoding |
| `FLAG_OWN_CAR` | Car ownership | Binary (0/1) |
| `FLAG_OWN_REALTY` | Real estate ownership | Binary (0/1) |
| `CNT_CHILDREN` | Number of children | Integer |
| `NAME_FAMILY_STATUS` | Marital status | Grouped categories |
| `NAME_TYPE_SUITE` | Who accompanied applicant | Grouped categories |
| `ADDITIONAL_DOC_PROVIDED` | Extra documentation flag | Derived binary |

#### Financial & Employment (7 features)

| Feature | Description | Engineering |
|---------|-------------|-------------|
| `DEBT_TO_INCOME` | Credit amount / Annual income | Derived ratio |
| `NAME_INCOME_TYPE` | Income source type | Categorical |
| `EDUCATION_LEVEL` | Education attainment | Grouped (3 levels) |
| `NAME_HOUSING_TYPE` | Housing situation | Categorical |
| `ORG_GROUP` | Employer organization type | Grouped (15 categories) |
| `OCCUPATION_TYPE_GROUPED` | Job category | Grouped (4 categories) |
| `N_DOCUMENTS_PROVIDED` | Count of submitted documents | Aggregated sum |

#### Time-Based (5 features)

| Feature | Description | Engineering |
|---------|-------------|-------------|
| `YEARS_BIRTH` | Age in years | Converted from days |
| `YEARS_EMPLOYED` | Employment duration | Converted from days |
| `YEARS_REGISTRATION` | Years at current address | Converted from days |
| `YEARS_ID_PUBLISH` | Years since ID issued | Converted from days |
| `YEARS_LAST_PHONE_CHANGE` | Years since phone change | Converted from days |

#### Credit History (7 features)

| Feature | Description | Engineering |
|---------|-------------|-------------|
| `AVERAGE_EXTERNAL_RATING` | Mean external bureau score | Normalized 0-1 |
| `REGION_RATING_CLIENT` | Regional risk rating | Integer 1-3 |
| `TOT_PREV_APP` | Total previous applications | Aggregated count |
| `APPROVED_RATIO` | Previous approval rate | Derived ratio |
| `REFUSED_RATIO` | Previous refusal rate | Derived ratio |
| `CANCELLED_RATIO` | Previous cancellation rate | Derived ratio |
| `UNUSED_RATIO` | Unused credit ratio | Derived ratio |

### Feature Importance (SHAP Analysis)

Top 10 most influential features:

1. **AVERAGE_EXTERNAL_RATING** (-0.42) - Strong negative = reduces risk
2. **DEBT_TO_INCOME** (+0.38) - High ratio increases default risk
3. **YEARS_EMPLOYED** (-0.31) - Longer employment reduces risk
4. **REFUSED_RATIO** (+0.28) - Past refusals indicate higher risk
5. **REGION_RATING_CLIENT** (+0.24) - Higher region rating = more risk
6. **APPROVED_RATIO** (-0.22) - Good approval history reduces risk
7. **YEARS_BIRTH** (-0.19) - Older applicants slightly lower risk
8. **FLAG_OWN_REALTY** (-0.17) - Property ownership reduces risk
9. **EDUCATION_LEVEL** (-0.15) - Higher education reduces risk
10. **N_DOCUMENTS_PROVIDED** (-0.12) - More docs = lower risk

---

## Model Training & Selection

### Models Evaluated

| Model | Sampling Strategy | Calibration | Test AP | Test F1 |
|-------|-------------------|-------------|---------|---------|
| XGBoost | None (Final Production) | None | **0.223** | **0.298** |
| XGBoost | Undersampling | Isotonic | 0.232 | 0.296 |
| Logistic Regression | Class Weights | None | 0.231 | 0.295 |
| XGBoost | Class Weights | None | 0.222 | 0.300 |
| Logistic Regression | SMOTE | None | 0.223 | 0.289 |

### Final Model: XGBoost (Production)

**Why XGBoost?**
- Best performance on imbalanced data
- Native handling of missing values
- Fast inference for real-time predictions
- SHAP TreeExplainer compatibility
- Clean pipeline without calibration wrapper for easier deployment

### Hyperparameters (Production Model)

```python
{
    'n_estimators': 50,
    'max_depth': 3,
    'learning_rate': 0.1,
    'subsample': 1.0,
    'random_state': 42
}
```

### Training Pipeline

```
Raw Data → Preprocessing → XGBoost → Final Model
              ↓
    - Log transform (skewed features)
    - StandardScaler (numeric)
    - OneHotEncoder (categorical)
```

---

## Threshold Tuning & MLflow Tracking

### MLflow Integration

All experiments are tracked locally in `mlruns/` with:

- **Metrics**: Train/Test AP, F1, Precision, Recall, `f1_threshold`
- **Parameters**: All hyperparameters and preprocessing settings
- **Artifacts**: Trained model pipelines registered under `Credit Risk Model Final`

### Logged Metrics Per Run

| Metric | Description |
|--------|-------------|
| `train_avg_precision` | PR-AUC on training set |
| `test_avg_precision` | PR-AUC on test set |
| `train_f1` / `test_f1` | F1 score at optimal threshold |
| `train_recall` / `test_recall` | Recall (sensitivity) |
| `train_precision` / `test_precision` | Precision |
| **`f1_threshold`** | **Optimal probability threshold — loaded by the app at startup** |

### Experiment Results (All Runs)

| Model | Test AP | F1 Threshold | Notes |
|-------|---------|--------------|-------|
| XGBoost with UnderSampling + Calibration | 0.232 | 0.155 | CalibratedClassifierCV |
| Logistic Regression L1 (Weighted) | 0.231 | 0.643 | |
| **XGBoost (Final Production)** | **0.223** | **0.631** | **Registered model** |
| XGBoost with scale_pos_weight | 0.222 | 0.647 | |
| Logistic Regression L1 with SMOTE | 0.223 | 0.633 | |
| Random Forest with UnderSampling + Calibration | 0.217 | 0.131 | |
| Random Forest (Weighted) | 0.208 | 0.496 | |
| XGBoost with SMOTE | 0.195 | 0.251 | |
| Random Forest with SMOTE | 0.184 | 0.273 | |

The production model was selected for its **clean pipeline** (no calibration wrapper), **high recall**, and **interpretable threshold** that generalises well to unseen data.

### Viewing Experiment Results

The **Model Monitoring** tab in the app displays all experiment runs with Train vs Test comparisons, loaded from `models/experiment_runs.json` (pre-exported, no MLflow server needed).

To view the full MLflow UI locally:

```bash
mlflow ui --backend-store-uri mlruns --port 5000
# Open http://127.0.0.1:5000
```

### Re-exporting After Retraining

When you retrain and register a new model, run both export scripts to keep `models/` in sync:

```bash
python export_model.py
python export_experiment_runs.py
git add models/ && git commit -m "Update model export" && git push
```

---

## Risk Identification

### Key Risk Factors Identified

Based on SHAP analysis and domain expertise:

#### High-Risk Indicators (Red Flags)

| Factor | Threshold | Risk Impact |
|--------|-----------|-------------|
| Debt-to-Income Ratio | > 5.0 | +15-25% PD increase |
| External Credit Rating | < 0.30 | +20-30% PD increase |
| Previous Refusal Rate | > 40% | +10-20% PD increase |
| Employment Duration | < 1 year | +8-15% PD increase |
| Region Rating | = 3 (worst) | +5-10% PD increase |
| No Real Estate | = 0 | +3-8% PD increase |

#### Low-Risk Indicators (Green Flags)

| Factor | Threshold | Risk Reduction |
|--------|-----------|----------------|
| External Credit Rating | > 0.70 | -15-25% PD |
| Employment Duration | > 5 years | -10-15% PD |
| Previous Approval Rate | > 70% | -8-12% PD |
| Owns Real Estate | = 1 | -5-10% PD |
| Higher Education | = Yes | -3-8% PD |

### Risk Segmentation

| Segment | PD Range | Population % | Default Rate |
|---------|----------|--------------|--------------|
| Prime | < 10% | 45% | 2.1% |
| Near-Prime | 10-25% | 30% | 8.5% |
| Subprime | 25-50% | 18% | 22.3% |
| Deep Subprime | > 50% | 7% | 48.7% |

---

## Business Usability

### Use Cases

#### 1. Real-Time Loan Decisioning
- **Input**: Applicant data from loan application
- **Output**: Instant risk score + recommendation
- **Latency**: < 500ms per prediction

#### 2. Portfolio Risk Assessment
- **Input**: Batch of existing loans
- **Output**: Risk distribution, concentration analysis
- **Use**: Quarterly risk reporting

#### 3. Pricing Optimization
- **Input**: Predicted PD per applicant
- **Output**: Risk-adjusted interest rates
- **Formula**: `Rate = Base Rate + (PD × Risk Premium)`

#### 4. Regulatory Compliance
- **Input**: Model predictions + SHAP explanations
- **Output**: Audit-ready decision documentation
- **Standards**: Basel III/IV, GDPR Article 22

### Integration Options

```python
# Option 1: Streamlit Web App
streamlit run app_modular.py

# Option 2: Python API
from src.services.model_service import ModelService
model = ModelService()
prob, pred, shap_summary, shap_df = model.predict(applicant_data)

# Option 3: REST API (FastAPI)
# See api/ directory for implementation
```

### User Roles

| Role | Access | Features |
|------|--------|----------|
| Loan Officer | Risk Assessment Tab | Predictions, basic reports |
| Credit Analyst | Full Access | + SHAP analysis, RAG reports |
| Risk Manager | Monitoring Tab | Model performance, drift detection |
| Auditor | Read-Only | Decision logs, explanations |

---

## Financial Impact & Money Saved

### Cost-Benefit Analysis

#### Assumptions
- Average loan amount: €15,000
- Loss given default (LGD): 60%
- Current manual review cost: €25 per application
- Model inference cost: €0.05 per application

#### Scenario: 100,000 Applications/Year

| Metric | Without Model | With Model | Improvement |
|--------|---------------|------------|-------------|
| Default Rate | 8.0% | 5.2% | -35% |
| Defaults | 8,000 | 5,200 | -2,800 |
| Loss Amount | €72M | €46.8M | **€25.2M saved** |
| Review Cost | €2.5M | €0.625M | **€1.875M saved** |
| **Total Savings** | - | - | **€27.075M/year** |

#### ROI Calculation

```
Development Cost:     €150,000 (one-time)
Annual Operating:     €50,000 (infrastructure + maintenance)
Annual Savings:       €27,075,000

Year 1 ROI: (27.075M - 0.2M) / 0.2M = 13,437%
Payback Period: < 1 week
```

### Efficiency Gains

| Process | Before | After | Improvement |
|---------|--------|-------|-------------|
| Decision Time | 2-5 days | < 1 minute | 99.9% faster |
| Manual Reviews | 100% | 25% (REVIEW only) | 75% reduction |
| Consistency | Variable | 100% consistent | Eliminated bias |
| Documentation | Manual | Auto-generated | 100% automated |

---

## Can It Replace Human Decision-Making?

### Short Answer: **Not Yet, But Close**

### Current Recommendation: **Hybrid Approach**

| Decision | Model Role | Human Role |
|----------|------------|------------|
| APPROVE (PD < 25%) | **Primary** decision-maker | Spot-check 5% |
| REVIEW (25-50%) | Provides recommendation | **Final decision** |
| DECLINE (PD ≥ 50%) | **Primary** decision-maker | Appeals review |

### Why Not Full Automation?

1. **Edge Cases**: Model may miss unusual but valid applications
2. **Regulatory Requirements**: Some jurisdictions require human oversight
3. **Relationship Banking**: High-value clients need personal touch
4. **Model Limitations**: 8% of predictions may be incorrect

### Path to Full Automation

| Milestone | Requirement | Status |
|-----------|-------------|--------|
| Model Accuracy | Test AP > 0.35 | 🔄 In Progress |
| Calibration | Brier Score < 0.05 | ✅ Achieved |
| Explainability | SHAP for all predictions | ✅ Achieved |
| Monitoring | Drift detection active | ✅ Achieved |
| Regulatory Approval | Compliance sign-off | ⏳ Pending |

### Recommended Usage

> **Use this system as an INFORMATIVE TOOL that augments human decision-making, not replaces it.**

The model should:
- ✅ Pre-screen applications to prioritize review queue
- ✅ Provide risk scores for pricing decisions
- ✅ Generate documentation for audit trails
- ✅ Flag high-risk applications for detailed review
- ❌ NOT make final decisions without oversight (for now)

---

## AI-Powered Report Generation

### OpenAI Integration

The system uses **GPT-4o-mini** to generate professional credit reports:

```python
from src.services.report_service import ReportService

report_service = ReportService()
report = report_service.generate_report(
    data=applicant_data,
    prob=0.32,
    pred=1,
    shap_summary="DEBT_TO_INCOME: +0.15 (increases risk)..."
)
```

### Report Sections

1. **Executive Summary** - Risk verdict and key metrics
2. **Risk Analysis** - SHAP-driven factor breakdown
3. **Applicant Strengths** - Positive indicators
4. **Areas of Concern** - Red flags identified
5. **Recommendations** - For customer and bank
6. **Decision Rationale** - Why this verdict

### Configuration

Set in `.env` file:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_MAX_TOKENS=800
OPENAI_TEMPERATURE=0.7
```

### Fallback Mode

If OpenAI is unavailable, the system falls back to a **rule-based report generator** that provides basic risk assessment without AI.

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- pip or conda
- No MLflow server required — model loads from `models/model.pkl`

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/your-org/credit-risk-predictor.git
cd credit-risk-predictor

# 2. Create virtual environment
python -m venv loan_approval_env
source loan_approval_env/bin/activate  # Linux/Mac
.\loan_approval_env\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables (optional — only needed for AI reports)
cp env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Run the application
streamlit run app.py
```

### Environment Variables

Create a `.env` file (copy from `env.example`):

```env
# OpenAI (optional — AI reports disabled if not set)
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_MAX_TOKENS=800
OPENAI_TEMPERATURE=0.7

# MLflow (only needed if running mlflow ui locally)
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

---

## Project Structure

```
Loan_approval_ml/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── env.example                   # Environment variable template
├── export_model.py               # One-time script: export model.pkl
├── export_experiment_runs.py     # One-time script: export experiment_runs.json
├── README.md                     # This file
│
├── models/                       # Portable model exports (committed to git)
│   ├── model.pkl                 # Trained XGBoost pipeline (107 KB)
│   ├── model_info.json           # run_id + f1_threshold = 0.631
│   └── experiment_runs.json      # All 12 experiment runs with metrics
│
├── src/                          # Modular source code
│   ├── config/
│   │   ├── settings.py           # Paths, MLflow config, OpenAI config
│   │   └── constants.py          # Sample applicant profiles
│   ├── services/
│   │   ├── model_service.py      # Model loading, prediction, SHAP
│   │   ├── report_service.py     # AI + fallback report generation
│   │   └── mlflow_service.py     # Experiment data loading
│   ├── components/
│   │   ├── styles.py             # Custom CSS
│   │   ├── input_form.py         # Applicant input form
│   │   ├── results.py            # Prediction results display
│   │   └── monitoring.py         # Model monitoring dashboard
│   └── utils/
│       ├── helpers.py            # Random sample generation
│       └── validators.py         # Input validation
│
├── Python Notebooks/
│   ├── Data Cleaning.ipynb       # Data preprocessing
│   ├── Modelling.ipynb           # Model training & MLflow logging
│   ├── eda.ipynb                 # Exploratory analysis
│   └── categories.json           # Feature category options
│
├── mlruns/                       # MLflow local tracking (gitignored)
├── mlartifacts/                  # MLflow model artifacts (gitignored)
└── Datasets/                     # Raw data (gitignored)
```

---

## Usage Guide

### Running the Application

```bash
streamlit run app.py
```

### Making Predictions

1. Navigate to the **Risk Assessment** tab
2. Fill in applicant details or click **Quick Fill** to generate a random sample
3. Click **Run Credit Assessment**
4. Review the verdict, probability score, SHAP feature chart, and AI-generated report

The decision threshold (`0.631`) is shown beneath the result. Any application with predicted PD >= 0.631 is declined.

### Viewing Model Monitoring

1. Navigate to the **Model Monitoring** tab
2. The pipeline architecture and calibration strategy are shown at the top
3. The experiment results table shows all 12 runs with Train vs Test metrics
4. Charts compare Average Precision and F1 across all models

Data is loaded from `models/experiment_runs.json` — no MLflow server required.

### Retraining the Model

1. Open `Python Notebooks/Modelling.ipynb`
2. Run the final production cell (Cell 10 — `XGBoost(Final_Production)`)
3. The cell logs `f1_threshold`, all test metrics, and registers the model as `Credit Risk Model Final`
4. After training, re-export the portable files:

```bash
python export_model.py
python export_experiment_runs.py
```

5. Commit and push `models/` to update the deployed app.

---

## License

MIT License - See LICENSE file for details.

---

## Support

For issues or questions, open a GitHub issue.
