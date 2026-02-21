"""
One-time script: exports all experiment runs from mlruns/ to
models/experiment_runs.json so the monitoring dashboard works
on Streamlit Cloud without a live MLflow server.

Run once from the project root:
    python export_experiment_runs.py
"""
import json
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()
MLRUNS_DIR   = PROJECT_ROOT / "mlruns"
DEST_FILE    = PROJECT_ROOT / "models" / "experiment_runs.json"

EXPERIMENT_NAME = "Credit Risk Modelling Project v6"

METRICS_TO_EXPORT = [
    "f1_threshold",
    "train_avg_precision",
    "train_f1",
    "train_precision",
    "train_recall",
    "test_avg_precision",
    "test_f1",
    "test_precision",
    "test_recall",
]


def read_metric(metric_file: pathlib.Path) -> float | None:
    """Read a single scalar value from an MLflow metric file."""
    try:
        parts = metric_file.read_text().strip().split()
        return float(parts[1]) if len(parts) >= 2 else float(parts[0])
    except Exception:
        return None


def read_tag(tag_file: pathlib.Path) -> str:
    try:
        return tag_file.read_text().strip()
    except Exception:
        return ""


def find_experiment_id(mlruns_dir: pathlib.Path, exp_name: str) -> str | None:
    for d in mlruns_dir.iterdir():
        if d.is_dir() and d.name.isdigit():
            meta = d / "meta.yaml"
            if meta.exists() and exp_name in meta.read_text():
                return d.name
    return None


def export_runs():
    exp_id = find_experiment_id(MLRUNS_DIR, EXPERIMENT_NAME)
    if not exp_id:
        print(f"Experiment '{EXPERIMENT_NAME}' not found in mlruns/")
        return

    exp_dir = MLRUNS_DIR / exp_id
    runs = []

    for rd in exp_dir.iterdir():
        if not (rd.is_dir() and len(rd.name) == 32):
            continue

        run_name = read_tag(rd / "tags" / "mlflow.runName") or rd.name[:8]

        metrics = {}
        for m in METRICS_TO_EXPORT:
            val = read_metric(rd / "metrics" / m)
            if val is not None:
                metrics[m] = round(val, 6)

        # Only include runs that have at least test_avg_precision
        if "test_avg_precision" not in metrics:
            continue

        # Read params
        params = {}
        params_dir = rd / "params"
        if params_dir.exists():
            for pf in params_dir.iterdir():
                try:
                    params[pf.name] = pf.read_text().strip()
                except Exception:
                    pass

        runs.append({
            "run_id":   rd.name,
            "run_name": run_name,
            "metrics":  metrics,
            "params":   params,
        })

    # Sort by test_avg_precision descending
    runs.sort(key=lambda r: r["metrics"].get("test_avg_precision", 0), reverse=True)

    DEST_FILE.parent.mkdir(exist_ok=True)
    DEST_FILE.write_text(json.dumps({"experiment_name": EXPERIMENT_NAME, "runs": runs}, indent=2))
    print(f"Exported {len(runs)} runs to {DEST_FILE}")
    for r in runs:
        print(f"  {r['run_name']:<45} test_ap={r['metrics'].get('test_avg_precision', 'N/A'):.4f}  f1_thresh={r['metrics'].get('f1_threshold', 'N/A')}")


if __name__ == "__main__":
    export_runs()
