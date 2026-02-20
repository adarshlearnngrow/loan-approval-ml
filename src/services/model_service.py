"""
Model Service - Handles model loading, prediction, and SHAP explanations
"""
import json
import pandas as pd
import streamlit as st
import mlflow.sklearn

from ..config.settings import (
    PROJECT_ROOT, MLFLOW_MODEL_NAME,
    MLFLOW_EXPERIMENT_NAME, CATEGORIES_FILE,
    MODEL_PKL_PATH, MODEL_INFO_PATH
)


# ── Module-level cached functions (must be outside the class so Streamlit
# ── never tries to hash 'self' as a cache key argument)

@st.cache_resource(show_spinner="Loading model...")
def _load_model_cached():
    """Load model and optimal threshold (cached).

    Strategy (in order):
    1. models/model.pkl + models/model_info.json  — portable, works everywhere
    2. mlruns/ + mlartifacts/                     — local dev with MLflow only

    Returns (model, threshold, run_id).
    """
    import pickle

    # ── Strategy 1: exported model.pkl (portable) ─────────────────────────────
    if MODEL_PKL_PATH.exists() and MODEL_INFO_PATH.exists():
        try:
            with open(MODEL_PKL_PATH, "rb") as f:
                model = pickle.load(f)
            info = json.loads(MODEL_INFO_PATH.read_text())
            threshold = float(info.get("f1_threshold", 0.5))
            run_id = info.get("run_id", "exported")
            return model, threshold, run_id
        except Exception as e:
            st.warning(f"Could not load exported model.pkl: {e}. Trying mlruns/...")

    # ── Strategy 2: local mlruns/ + mlartifacts/ ──────────────────────────────
    try:
        mlruns_dir = PROJECT_ROOT / "mlruns"
        mlartifacts_dir = PROJECT_ROOT / "mlartifacts"

        if not mlruns_dir.exists():
            raise FileNotFoundError("mlruns/ directory not found")

        # Read registered model meta.yaml for exact run_id + model_id
        model_registry_dir = mlruns_dir / "models" / MLFLOW_MODEL_NAME
        if not model_registry_dir.exists():
            raise FileNotFoundError(
                f"Registered model '{MLFLOW_MODEL_NAME}' not found in mlruns/models/"
            )

        version_dirs = sorted(
            [d for d in model_registry_dir.iterdir()
             if d.is_dir() and d.name.startswith("version-")],
            key=lambda d: int(d.name.split("-")[1]),
            reverse=True
        )
        if not version_dirs:
            raise FileNotFoundError(f"No versions found for '{MLFLOW_MODEL_NAME}'")

        meta_text = (version_dirs[0] / "meta.yaml").read_text()
        run_id = model_id = exp_id = None
        for line in meta_text.splitlines():
            if line.startswith("run_id:"):
                run_id = line.split(":", 1)[1].strip()
            elif line.startswith("model_id:"):
                model_id = line.split(":", 1)[1].strip()
            elif line.startswith("storage_location:"):
                loc = line.split(":", 2)[-1].strip().lstrip("/")
                exp_id = loc.split("/")[0]

        if not run_id or not model_id or not exp_id:
            raise ValueError("Could not parse run_id/model_id/exp_id from meta.yaml")

        # Read f1_threshold
        threshold = 0.5
        thresh_file = mlruns_dir / exp_id / run_id / "metrics" / "f1_threshold"
        if thresh_file.exists():
            parts = thresh_file.read_text().strip().split()
            threshold = float(parts[1]) if len(parts) >= 2 else float(parts[0])

        # Load model from mlartifacts
        model_path = mlartifacts_dir / exp_id / "models" / model_id / "artifacts"
        if not (model_path / "MLmodel").exists():
            raise FileNotFoundError(f"MLmodel not found at {model_path}")

        model = mlflow.sklearn.load_model(str(model_path))
        return model, threshold, run_id

    except Exception as exc:
        st.error(f"Model could not be loaded: {exc}")
        return None, 0.5, "Unknown"


@st.cache_resource
def _get_shap_explainer(_base_model):
    """Build and cache the SHAP TreeExplainer once per session."""
    import shap
    return shap.TreeExplainer(_base_model)


@st.cache_data
def _load_categories_cached():
    """Load feature categories from JSON (cached)"""
    if CATEGORIES_FILE.exists():
        return json.load(open(CATEGORIES_FILE))
    return {}


class ModelService:
    """Service for ML model operations"""

    def __init__(self):
        self._model = None
        self._threshold = 0.5
        self._run_id = "Unknown"
        self._categories = None

    def get_model(self):
        """Get the loaded model and its optimal decision threshold"""
        if self._model is None:
            self._model, self._threshold, self._run_id = _load_model_cached()
        return self._model

    def get_categories(self):
        """Get feature categories"""
        if self._categories is None:
            self._categories = _load_categories_cached()
        return self._categories
    
    @property
    def threshold(self) -> float:
        """Optimal decision threshold retrieved from MLflow run metrics (f1_threshold)"""
        if self._model is None:
            self.get_model()  # triggers load
        return self._threshold

    @property
    def run_id(self) -> str:
        """The specific MLflow Run ID that provided the current model & threshold"""
        if self._model is None:
            self.get_model()
        return self._run_id

    def predict(self, input_data: dict) -> tuple:
        """
        Make prediction on input data using the MLflow-logged optimal threshold.

        Returns:
            tuple: (probability, prediction, shap_summary, shap_df)
                   prediction is derived from prob vs the logged f1_threshold
        """
        model = self.get_model()
        df_input = pd.DataFrame([input_data])

        prob = float(model.predict_proba(df_input)[0, 1])
        pred = int(prob >= self._threshold)  # use logged threshold, not model default

        # Calculate SHAP values
        shap_summary, shap_df = self._calculate_shap(model, df_input)

        return prob, pred, shap_summary, shap_df
    
    def _calculate_shap(self, model, df_input):
        """Calculate SHAP values for model explanation"""
        shap_summary = ""
        shap_df = None
        
        try:
            import shap

            base = self._unwrap_xgboost(model)
            preproc = self._get_preproc_pipeline(model)
            
            is_xgb = type(base).__name__ == "XGBClassifier"
            if is_xgb:
                X_t = preproc.transform(df_input) if preproc else df_input.values
                try:
                    feat_names = preproc.get_feature_names_out() if preproc else list(df_input.columns)
                except Exception:
                    feat_names = [f"f{i}" for i in range(X_t.shape[1])]
                
                explainer = _get_shap_explainer(base)
                shap_vals = explainer.shap_values(X_t)
                
                # Target the last class (Class 1 = Default) for explanation
                if isinstance(shap_vals, list):
                    v = shap_vals[-1]
                else:
                    v = shap_vals
                
                # Create DataFrame for plotting
                shap_df = pd.DataFrame({
                    "Feature": feat_names,
                    "SHAP": v[0],
                }).reindex(pd.Series(v[0]).abs().sort_values(ascending=False).index).head(12)
                
                # String summary for LLM
                summary_lines = []
                for _, r in shap_df.head(8).iterrows():
                    direction = "increases risk" if r['SHAP'] > 0 else "reduces risk"
                    summary_lines.append(f"- {r['Feature']}: {r['SHAP']:.4f} ({direction})")
                shap_summary = "\n".join(summary_lines)
                
        except Exception:
            pass
        
        return shap_summary, shap_df
    
    @staticmethod
    def _unwrap_xgboost(m):
        """Unwrap calibrated/pipeline to get raw XGBClassifier"""
        if hasattr(m, "calibrated_classifiers_"):
            return ModelService._unwrap_xgboost(m.calibrated_classifiers_[0].estimator)
        if hasattr(m, "steps"):
            # Walk all steps and return the first one that looks like a classifier
            for _, step in reversed(m.steps):
                result = ModelService._unwrap_xgboost(step)
                if type(result).__name__ == "XGBClassifier":
                    return result
            return ModelService._unwrap_xgboost(m.steps[-1][1])
        if hasattr(m, "named_steps"):
            return ModelService._unwrap_xgboost(list(m.named_steps.values())[-1])
        return m
    
    @staticmethod
    def _get_preproc_pipeline(m):
        """Return a sklearn Pipeline of transformer-only steps (no samplers, no classifiers)."""
        if hasattr(m, "calibrated_classifiers_"):
            return ModelService._get_preproc_pipeline(m.calibrated_classifiers_[0].estimator)
        if hasattr(m, "steps"):
            from sklearn.pipeline import Pipeline as _P
            transformer_steps = [
                (name, step) for name, step in m.steps
                if hasattr(step, "transform")
                and not hasattr(step, "predict_proba")   # exclude classifiers
                and not hasattr(step, "fit_resample")    # exclude samplers (imblearn)
            ]
            return _P(transformer_steps) if transformer_steps else None
        return None
