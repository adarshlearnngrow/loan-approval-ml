"""
Application Settings and Configuration
"""
import os
import pathlib
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()

# Load environment variables from ROOT explicitly
load_dotenv(PROJECT_ROOT / ".env")

SRC_DIR = PROJECT_ROOT / "src"
NB_DIR = PROJECT_ROOT / "Python Notebooks"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
DATASETS_DIR = PROJECT_ROOT / "Datasets"
DOCUMENTS_DIR = PROJECT_ROOT / "Documents"

# ── MLflow Configuration ───────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000").strip().rstrip("\\/")
MLFLOW_MODEL_NAME = "Credit Risk Model Final"
MLFLOW_EXPERIMENT_NAME = "Credit Risk Modelling Project v6"

# ── OpenAI Configuration ───────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", 800))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.7))

# ── Application Settings ───────────────────────────────────────────────────────
APP_TITLE = "Credit Risk Predictor"
APP_SUBTITLE = "AI-powered credit decisioning and model performance monitoring"
APP_VERSION = "2.0.0"

# ── Risk Thresholds (Defaults, overwritten by MLflow if available) ──────────────
RISK_THRESHOLD_DEFAULT = 0.5

# ── Feature Configuration ──────────────────────────────────────────────────────
CATEGORIES_FILE = NB_DIR / "categories.json"

# ── Exported Model (portable fallback, no mlruns/ needed) ──────────────────────
MODEL_PKL_PATH         = PROJECT_ROOT / "models" / "model.pkl"
MODEL_INFO_PATH        = PROJECT_ROOT / "models" / "model_info.json"
EXPERIMENT_RUNS_PATH   = PROJECT_ROOT / "models" / "experiment_runs.json"
