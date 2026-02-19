"""
MLflow Service - Handles experiment tracking and model monitoring
"""
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from ..config.settings import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME


class MLflowService:
    """Service for MLflow operations and model monitoring"""
    
    def __init__(self):
        self._client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize MLflow client"""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self._client = MlflowClient()
    
    def get_experiment_runs(self, max_results: int = 20) -> pd.DataFrame:
        """
        Fetch experiment runs with metrics
        
        Returns:
            DataFrame with model performance metrics
        """
        all_exps = self._client.search_experiments()
        exp_ids = [e.experiment_id for e in all_exps if e.name == MLFLOW_EXPERIMENT_NAME]
        
        if not exp_ids:
            return pd.DataFrame()
        
        runs = self._client.search_runs(
            experiment_ids=exp_ids,
            order_by=["metrics.test_avg_precision DESC"],
            max_results=max_results,
        )
        
        rows = []
        for r in runs:
            m = r.data.metrics
            name = r.data.tags.get("mlflow.runName", r.info.run_id[:8])
            
            train_ap = m.get("train_avg_precision", m.get("average_precision_score", None))
            test_ap = m.get("test_avg_precision", None)
            
            if test_ap is None:
                continue
            
            gap = round(train_ap - test_ap, 3) if train_ap is not None else None
            run_url = f"{MLFLOW_TRACKING_URI}/#/experiments/{exp_ids[0]}/runs/{r.info.run_id}"
            
            rows.append({
                "Model": name,
                "Train AP": round(train_ap, 3) if train_ap is not None else None,
                "Test AP": round(test_ap, 3),
                "AP Gap": gap,
                "Test Precision": round(m.get("test_precision", 0), 3) if m.get("test_precision") else None,
                "Test Recall": round(m.get("test_recall", 0), 3) if m.get("test_recall") else None,
                "Test F1": round(m.get("test_f1", 0), 3) if m.get("test_f1") else None,
                "F1 Threshold": round(m.get("f1_threshold", 0), 3) if m.get("f1_threshold") else None,
                "Train F1": round(m.get("train_f1", 0), 3) if m.get("train_f1") else None,
                "Train Recall": round(m.get("train_recall", 0), 3) if m.get("train_recall") else None,
                "Train Precision": round(m.get("train_precision", 0), 3) if m.get("train_precision") else None,
                "View": f"[Inspect]({run_url})",
                "Run ID": r.info.run_id[:8],
            })
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        # Force metric columns to float64
        metric_cols = [
            "Train AP", "Test AP", "AP Gap", "Test Precision", "Test Recall",
            "Test F1", "F1 Threshold", "Train F1", "Train Recall", "Train Precision"
        ]
        for col in metric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df
    
    def get_best_model_info(self, df: pd.DataFrame) -> dict:
        """Get information about the best performing model"""
        if df.empty:
            return {}
        
        best = df.iloc[0]
        gap = best.get("AP Gap", 0)
        
        return {
            "name": best["Model"],
            "test_ap": best["Test AP"],
            "test_f1": best.get("Test F1"),
            "test_recall": best.get("Test Recall"),
            "ap_gap": gap,
            "generalization": "good" if isinstance(gap, float) and gap < 0.1 else "concerning"
        }
